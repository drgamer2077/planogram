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
import shutil

# Streamlit interface
st.title("Compliance Report Generator")
st.write("This code will execute itself at 03:00 AM IST every day to generate the compliance report!")

# Paths
images_folder = "../../home/site/wwwroot/Images"
images_old_folder = "../../home/site/wwwroot/Images_OLD"
bev_master_file_path = "../../home/site/wwwroot/Data/master_file.xlsx"
json_folder = "../../home/site/wwwroot/JSON"
report_folder = "../../home/site/wwwroot/Report"

# Create folders if they don't exist
for folder in [images_folder, json_folder, "Data", report_folder, images_old_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Roboflow API key and model ID
api_key = "YxeULFmRqt8AtNbwzXrT"
model_id = "cooler-image"
model_version = "1"

def calculate_chilled_uf_score(pack_compliance_output):
    keywords = ['Coke_small', 'Limca_small', 'thumpsup_small', 
                'Sprite_small', 'thumpsup_medium', 'Sprite_Big']
    
    chilled_scores = {}
    for image_id in pack_compliance_output['Image_id'].unique():
        image_data = pack_compliance_output[pack_compliance_output['Image_id'] == image_id]
        unique_matches = set(image_data[image_data['class'].isin(keywords)]['class'])
        chilled_scores[image_id] = len(unique_matches)
    
    return pd.Series(chilled_scores)

def calculate_purity_rcs(pack_compliance_output):
    keywords = ["Pepsi", "Mountain", "7up", "Slice", "Sting"]
    purity_scores = {}

    for image_id in pack_compliance_output['Image_id'].unique():
        image_data = pack_compliance_output[pack_compliance_output['Image_id'] == image_id]

        # Top shelves (shelf 1 and 2)
        top_shelves_data = image_data[image_data['shelf'].isin([1, 2])]
        total_top_shelves = len(top_shelves_data)
        if total_top_shelves > 0:
            top_keywords_count = top_shelves_data['class'].str.startswith(tuple(keywords)).sum()
            top_percentage = (top_keywords_count / total_top_shelves) * 100
            if top_percentage == 0:
                top_points = 10
            elif top_percentage < 10:
                top_points = 8
            else:
                top_points = 0
        else:
            top_points = 0

        # Other shelves (shelf > 2)
        other_shelves_data = image_data[image_data['shelf'] > 2]
        total_other_shelves = len(other_shelves_data)
        if total_other_shelves > 0:
            other_keywords_count = other_shelves_data['class'].str.startswith(tuple(keywords)).sum()
            other_percentage = (other_keywords_count / total_other_shelves) * 100

            if 30 <= other_percentage <= 35:
                other_points = 1
            elif 26 <= other_percentage <= 29:
                other_points = 2
            elif 20 <= other_percentage <= 25:
                other_points = 3
            elif 15 <= other_percentage <= 19:
                other_points = 4
            elif other_percentage <= 14:
                other_points = 5
            else:
                other_points = 0
        else:
            other_points = 0

        # Total Purity_RCS score for this Image_id
        purity_scores[image_id] = top_points + other_points

    return pd.Series(purity_scores)

def populate_outlet_data(report_df, outlet_master_folder):
    columns_to_fill = {
        "OTYP": "MainChannelType",
        "OUTLET_NM": "OutletName",
        "ASM": "ASM_Name",
        "SE": "RSE_Name",
        "PSR": "PSR_Desc"
    }
    
    for col in columns_to_fill.keys():
        if col not in report_df.columns:
            report_df[col] = ''
            
    outlet_master_files = [f for f in os.listdir(outlet_master_folder) 
                          if f.startswith("OutletMaster_") and f.endswith(".csv")]
    
    outlet_master_combined = pd.DataFrame()
    for file in outlet_master_files:
        file_path = os.path.join(outlet_master_folder, file)
        try:
            temp_df = pd.read_csv(file_path, usecols=["Outletid"] + list(columns_to_fill.values()))
            outlet_master_combined = pd.concat([outlet_master_combined, temp_df], ignore_index=True)
        except Exception as e:
            st.error(f"Error reading file {file}: {e}")
    
    outlet_master_combined.drop_duplicates(subset='Outletid', inplace=True)
    
    for report_col, master_col in columns_to_fill.items():
        report_df[report_col] = report_df['OUTLET CODE'].map(
            outlet_master_combined.set_index('Outletid')[master_col])
    
    return report_df

def generate_compliance_report():
    uploaded_files = [os.path.join(images_folder, file) 
                     for file in os.listdir(images_folder) 
                     if file.endswith(('jpg', 'jpeg', 'png'))]
    
    bev_master_file = bev_master_file_path
    
    if not uploaded_files or not bev_master_file:
        st.error("Please upload all required files.")
        return
    
    rf = roboflow.Roboflow(api_key=api_key)
    model = rf.workspace().project(model_id).version(model_version).model
    
    def get_json_op(image_file_path):
        try:
            predictions_response = model.predict(image_file_path).json()
            predictions = predictions_response.get("predictions", [])
            output_json_path = os.path.join(json_folder, f"OP_{os.path.basename(image_file_path)}.json")
            with open(output_json_path, 'w') as json_file:
                json.dump(predictions, json_file, indent=4)
            return output_json_path
        except Exception as e:
            st.error(f"Error processing image {image_file_path}: {e}")
            return None
    
    json_paths = []
    for file in uploaded_files:
        json_path = get_json_op(file)
        if json_path:
            json_paths.append(json_path)
    
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
        try:
            df = pd.read_json(json_img_op)
            df = df.drop(columns='image_path', errors='ignore')
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
            df['pack_order_check'] = df.apply(
                lambda row: 1 if row['actual size (json op)'] != row['expected size'] else 0, axis=1)
            return df
        except Exception as e:
            st.error(f"Error processing JSON {json_img_op}: {str(e)}")
            return pd.DataFrame()
    
    def brand_order_comp(json_img_op):
        poc_op = pack_order_comp(json_img_op)
        if poc_op.empty:
            return pd.DataFrame()
        
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
        if not output_df.empty:
            pack_compliance_output = pd.concat([pack_compliance_output, output_df], ignore_index=True)
        
        brand_output_df = brand_order_comp(json_path)
        if not brand_output_df.empty:
            brand_compliance_df = pd.concat([brand_compliance_df, brand_output_df], ignore_index=True)
    
    pack_order_check = pd.DataFrame(
        pack_compliance_output.groupby('Image_id')['pack_order_check'].sum().reset_index())
    pack_order_check['pack_order_score'] = pack_order_check.apply(
        lambda row: 0 if row['pack_order_check'] > 0 else 2, axis=1)
    
    brand_order_check = pd.DataFrame(
        brand_compliance_df.groupby('Image_id')['brand_order_check'].sum().reset_index())
    brand_order_check['brand_order_score'] = brand_order_check.apply(
        lambda row: 3 if row['brand_order_check'] == 5 else 0, axis=1)
    
    final_op = pd.merge(pack_order_check, brand_order_check, on='Image_id', how='outer').fillna(0)
    
    # Calculate Chilled_UF_Scoring_RCS
    chilled_scores = calculate_chilled_uf_score(pack_compliance_output)
    final_op['Chilled_UF_Scoring_RCS'] = final_op['Image_id'].map(chilled_scores)

    # Calculate Purity_RCS scores and map them to the final DataFrame
    purity_scores = calculate_purity_rcs(pack_compliance_output)
    final_op['Purity_RCS'] = final_op['Image_id'].map(purity_scores)
    
    # Add new columns
    final_op['MONTH'] = datetime.now().strftime("%B")
    final_op['OTYP'] = ''
    final_op['OUTLET_NM'] = ''
    final_op['DL1_ADD1'] = ''
    final_op['VPO_CLASS'] = ''
    final_op['OUTLET CODE'] = final_op['Image_id'].apply(lambda x: x.split('_')[0])
    final_op['SM'] = ''
    final_op['ASM'] = ''
    final_op['SE'] = ''
    final_op['PSR'] = ''
    final_op['RT'] = ''
    final_op['Visible_Accessible_RCS'] = 0
    #final_op['Purity_RCS'] = 0
    final_op['Brand_Order_Compliance_Check'] = final_op['brand_order_check']
    final_op['Brand_Order_Compliance_RCS'] = final_op['brand_order_score']
    final_op['Pack_Order_Compliance_Test'] = final_op['pack_order_check']
    final_op['Pack_Order_Compliance_RCS'] = final_op['pack_order_score']
    
    final_op['Total_Equipment_Score'] = final_op[['Visible_Accessible_RCS', 
        'Purity_RCS', 'Chilled_UF_Scoring_RCS', 'Brand_Order_Compliance_RCS', 
        'Pack_Order_Compliance_RCS']].sum(axis=1)
    
    # Populate outlet data
    final_op = populate_outlet_data(final_op, report_folder)
    
    # Reorder columns
    columns_order = ['MONTH', 'OTYP', 'OUTLET_NM', 'DL1_ADD1', 'VPO_CLASS', 'OUTLET CODE', 
                    'SM', 'ASM', 'SE', 'PSR', 'RT', 'Visible_Accessible_RCS', 'Purity_RCS', 
                    'Chilled_UF_Scoring_RCS', 'Brand_Order_Compliance_Check', 
                    'Brand_Order_Compliance_RCS', 'Pack_Order_Compliance_Test', 
                    'Pack_Order_Compliance_RCS', 'Total_Equipment_Score']
    final_op = final_op[columns_order]
    
    # Save to Excel
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    compliance_report_path = os.path.join(report_folder, f"COMPLIANCE_REPORT_{current_time}.xlsx")
    
    with pd.ExcelWriter(compliance_report_path, engine='openpyxl') as writer:
        final_op.to_excel(writer, sheet_name='Compliance Scores', index=False)
        pack_compliance_output.to_excel(writer, sheet_name='Pack Order Compliance', index=False)
        brand_compliance_df.to_excel(writer, sheet_name='Brand Order Compliance', index=False)
    
    st.success("Compliance report generated successfully!")
    
    # Move processed images to Images_OLD folder
    for file in uploaded_files:
        try:
            shutil.move(file, os.path.join(images_old_folder, os.path.basename(file)))
        except Exception as e:
            st.warning(f"Could not move file {file} to {images_old_folder}: {e}")

def check_images_folder():
    while True:
        if not os.listdir(images_folder):
            st.warning("No images in the Images folder. Please upload images.")
        else:
            generate_compliance_report()
            break
        time.sleep(10)

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
