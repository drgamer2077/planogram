import pandas as pd
import json
import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import roboflow

app = Flask(__name__)

# Replace with your Roboflow API key and model ID
rf = roboflow.Roboflow(api_key="YxeULFmRqt8AtNbwzXrT")
model = rf.workspace().project("cooler-image").version("1").model

# Define upload folder and allowed extensions
UPLOAD_FOLDER = './Image Folder/'
JSON_OP_FOLDER = './JSON_ops/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_json_op(upload_folder, image_name):
    image_path = os.path.join(upload_folder, image_name)
    predictions_response = model.predict(image_path).json()  # Get the JSON response
    predictions = predictions_response.get("predictions", [])
    # Save predictions to a JSON file
    output_json_path = os.path.join(JSON_OP_FOLDER, f"OP_{image_name}.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(predictions, json_file, indent=4)
    return output_json_path

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

bev_master = pd.read_excel('./Data Files/Beverages_products_master.xlsx')

def pack_order_comp(json_img_op):
    df = pd.read_json(os.path.join(JSON_OP_FOLDER, json_img_op))
    df = df.drop(columns='image_path')
    df = df.sort_values(by=['y', 'x']).reset_index(drop=True)
    df['Image_id'] = json_img_op.strip('.json').strip('OP_')
    
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
    comparison_df['Image_id'] = json_img_op.strip('.json').strip('OP_')
    return comparison_df

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        json_op_path = get_json_op(app.config['UPLOAD_FOLDER'], filename)
        return jsonify({"message": "File uploaded successfully", "json_op_path": json_op_path}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/generate_report', methods=['GET'])
def generate_report():
    json_img_ops = os.listdir(JSON_OP_FOLDER)
    
    pack_compliance_output = pd.DataFrame()
    brand_compliance_df = pd.DataFrame()
    
    for i in json_img_ops:
        output_df = pack_order_comp(i)
        pack_compliance_output = pd.concat([pack_compliance_output, output_df], ignore_index=True)
        pack_compliance_output = pack_compliance_output[['Image_id', 'x', 'y', 'width', 'height', 'confidence', 'class', 'class_id',
                                                         'detection_id', 'prediction_type', 'shelf',
                                                         'actual size (json op)', 'flavour_type', 'expected size', 'pack_order_check']]
        
        brand_output_df = brand_order_comp(i)
        brand_compliance_df = pd.concat([brand_compliance_df, brand_output_df], ignore_index=True)
        brand_compliance_df = brand_compliance_df[['Image_id', 'Shelf', 'Flavour List', 'Ideal Order', 'brand_order_check']]
    
    pack_order_check = pd.DataFrame(pack_compliance_output.groupby('Image_id')['pack_order_check'].sum().reset_index())
    pack_order_check['pack_order_score'] = pack_order_check.apply(lambda row: 0 if row['pack_order_check'] > 0 else 2, axis=1)
    
    brand_order_check = pd.DataFrame(brand_compliance_df.groupby('Image_id')['brand_order_check'].sum().reset_index())
    brand_order_check['brand_order_score'] = brand_order_check.apply(lambda row: 3 if row['brand_order_check'] == 5 else 0, axis=1)
    
    final_op = pd.merge(pack_order_check, brand_order_check, on='Image_id')
    final_op = final_op.drop(columns=['pack_order_check', 'brand_order_check'])
    
    report_path = 'COMPLIANCE REPORT/COMPLIANCE REPORT.xlsx'
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        final_op.to_excel(writer, sheet_name='Compliance Scores', index=False)
        pack_compliance_output.to_excel(writer, sheet_name='Pack Order Compliance', index=False)
        brand_compliance_df.to_excel(writer, sheet_name='Brand Order Compliance', index=False)
    
    return send_file(report_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)