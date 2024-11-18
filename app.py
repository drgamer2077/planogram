import streamlit as st
import pandas as pd
import json
import roboflow
import os
import tempfile
import schedule
import time
from datetime import datetime, timedelta
import pytz

# Streamlit interface
st.title("Compliance Report Generator")
st.write("This code will execute itself at 03:00 AM IST every day to generate the compliance report!")

# Paths
images_folder = "../../home/site/wwwroot/Images"
bev_master_file_path = "../../home/site/wwwroot/Data/master_file.xlsx"
json_folder = "../../home/site/wwwroot/JSON"
report_folder = "../../home/site/wwwroot/Report"

# Create folders if they don't exist
for folder in [images_folder, json_folder, "Data", report_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Roboflow API key and model ID
api_key = "YxeULFmRqt8AtNbwzXrT"
model_id = "cooler-image"
model_version = "1"

def generate_compliance_report():
    uploaded_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(('jpg', 'jpeg', 'png'))]
    bev_master_file = bev_master_file_path

    if not uploaded_files or not bev_master_file:
        st.error("Please upload all required files.")
    else:
        img_names = [os.path.basename(file) for file in uploaded_files]
        
        rf = roboflow.Roboflow(api_key=api_key)
        model = rf.workspace().project(model_id).version(model_version).model

        def get_json_op(image_file_path):
            predictions_response = model.predict(image_file_path).json()  # Get the JSON response
            predictions = predictions_response.get("predictions", [])
            
            output_json_path = os.path.join(json_folder, f"OP_{os.path.basename(image_file_path)}.json")
            with open(output_json_path, 'w') as json_file:
                json.dump(predictions, json_file, indent=4)
            return output_json_path

        json_paths = []
        for file in uploaded_files:
            json_paths.append(get_json_op(file))

        def size_classification(name):
            if 'small' in name.lower():
                return "ic"
            elif 'medium' in name.lower():
                return "otg"
            elif 'big' in name.lower() or 'large' in name.lower():
                return "fc"
            else:
                return ""

        def follows_order(ideal_order, current_order):
            ideal_index = 0
            for item in current_order:
                while ideal_index < len(ideal_order) and ideal_order[ideal_index] != item:
                    ideal_index += 1
                if ideal_index == len(ideal_order):
                    return 0
            return 1

        def expected_shelf_op(shelf):
            if shelf in [1, 2]:
                return "ic"
            elif shelf in [3, 4]:
                return "otg"
            elif shelf >= 5:
                return "fc"
            else:
                return ""

        ideal_order = ['Cola', 'Flavour', 'Energy Drink', 'Stills', 'Mixers', 'Water']

        bev_master = pd.read_excel(bev_master_file)

        def pack_order_comp(json_img_op):
            df = pd.read_json(json_img_op)
            df = df.drop(columns='image_path')
            df = df.sort_values(by=['y', 'x']).reset_index(drop=True)
            df['Image_id'] = os.path.basename(json_img_op).strip('.json').strip('OP_')
            
            df['y_diff'] = df['y'].diff().fillna(0)
            threshold = 50
            df['new_bin'] = (df['y_diff'] > threshold).cumsum()
            df['shelf'] = df['new_bin'].apply(lambda x: f'{x+1}')
            df['shelf'] = df['shelf'].astype('int')
            df.drop(columns=['y_diff', 'new_bin'], inplace=True)
            df = df.sort_values(by=['shelf', 'x'])
            
            df['actual size (json op)'] = df['class'].apply(size_classification)
            df = pd.merge(df, bev_master[['class_id', 'flavour_type']], on='class_id', how='left')
            df['expected size'] = df['shelf'].apply(expected_shelf_op)
            df['pack_order_check'] = df.apply(lambda row: 1 if row['actual size (json op)'] != row['expected size'] else 0, axis=1)
            
            return df

        def brand_order_comp(json_img_op):
            poc_op = pack_order_comp(json_img_op)
            shelf_flavour_mapping = poc_op.groupby('shelf')['flavour_type'].apply(list).to_dict()
            comparison_result = []
            for shelf, flavours in shelf_flavour_mapping.items():
                result = {
                    'Shelf': shelf,
                    'Flavour List': flavours,
                    'Ideal Order': ideal_order,
                    'brand_order_check': follows_order(ideal_order, flavours)
                }
                comparison_result.append(result)
            
            comparison_df = pd.DataFrame(comparison_result)
            comparison_df['Image_id'] = os.path.basename(json_img_op).strip('.json').strip('OP_')
            return comparison_df

        pack_compliance_output = pd.DataFrame()
        brand_compliance_df = pd.DataFrame()

        for json_path in json_paths:
            output_df = pack_order_comp(json_path)
            pack_compliance_output = pd.concat([pack_compliance_output, output_df], ignore_index=True)
            pack_compliance_output = pack_compliance_output[['Image_id', 'x', 'y', 'width', 'height', 'confidence', 'class', 'class_id',
                                                             'detection_id', 'prediction_type', 'shelf',
                                                             'actual size (json op)', 'flavour_type', 'expected size', 'pack_order_check']]

            brand_output_df = brand_order_comp(json_path)
            brand_compliance_df = pd.concat([brand_compliance_df, brand_output_df], ignore_index=True)
            brand_compliance_df = brand_compliance_df[['Image_id', 'Shelf', 'Flavour List', 'Ideal Order', 'brand_order_check']]

        pack_order_check = pd.DataFrame(pack_compliance_output.groupby('Image_id')['pack_order_check'].sum().reset_index())
        pack_order_check['pack_order_score'] = pack_order_check.apply(lambda row: 0 if row['pack_order_check'] > 0 else 2, axis=1)

        brand_order_check = pd.DataFrame(brand_compliance_df.groupby('Image_id')['brand_order_check'].sum().reset_index())
        brand_order_check['brand_order_score'] = brand_order_check.apply(lambda row: 3 if row['brand_order_check'] == 5 else 0, axis=1)

        final_op = pd.merge(pack_order_check, brand_order_check, on='Image_id')
        final_op = final_op.drop(columns=['pack_order_check', 'brand_order_check'])

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        compliance_report_path = os.path.join(report_folder, f"COMPLIANCE_REPORT_{current_time}.xlsx")
        with pd.ExcelWriter(compliance_report_path, engine='openpyxl') as writer:
            final_op.to_excel(writer, sheet_name='Compliance Scores', index=False)
            pack_compliance_output.to_excel(writer, sheet_name='Pack Order Compliance', index=False)
            brand_compliance_df.to_excel(writer, sheet_name='Brand Order Compliance', index=False)

        st.success("Compliance report generated successfully!")

        # Delete all images in the images folder
        for file in uploaded_files:
            os.remove(file)

def check_images_folder():
    while True:
        if not os.listdir(images_folder):
            st.warning("No images in the Images folder. Please upload images.")
        else:
            generate_compliance_report()
            break
        time.sleep(10)  # Check every 10 seconds

# Schedule the task
schedule_time = "03:00"
ist = pytz.timezone('Asia/Kolkata')
utc = pytz.utc

def run_scheduled_task():
    now_utc = datetime.now(utc)
    now_ist = now_utc.astimezone(ist)
    target_time_ist = datetime.strptime(schedule_time, "%H:%M").time()
    target_datetime_ist = datetime.combine(now_ist.date(), target_time_ist)
    target_datetime_utc = ist.localize(target_datetime_ist).astimezone(utc)

    if now_utc >= target_datetime_utc:
        check_images_folder()
        target_datetime_utc += timedelta(days=1)

    delay = (target_datetime_utc - now_utc).total_seconds()
    time.sleep(delay)
    check_images_folder()

# Run the scheduler
while True:
    run_scheduled_task()
    time.sleep(1)
