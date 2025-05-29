import csv
import uuid
from flask import Flask, request, jsonify, render_template
from PIL import Image
from flask_cors import CORS
import json
from io import BytesIO
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
import torch
from flask import Flask, request, jsonify, send_from_directory
# from ultralytics import YOLO
import subprocess
import tempfile
import cv2
import numpy as np
# from logo_detection.logo_detection_module import yolov8_logo_detection
# from saliency_prediction.saliency_prediction_module import saliency_map_prediction_brand
# import pytesseract
import google.generativeai as genai
import time
import os
from paddleocr import PaddleOCR
import traceback
# from inference_sdk import InferenceHTTPClient
import torch
# from ultralytics import YOLO
from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
import cv2
import requests
import time
import datetime
import re
from gradio_client import Client, handle_file
import time
from PIL import Image
# import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import random
import warnings
import joblib
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import os
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.cluster import KMeans
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.measure import shannon_entropy  # For shape and texture analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# import matplotlib.pyplot as plt

import logging
from gradio_client import Client, file
from PIL import Image, ImageDraw
import cv2
import numpy as np
from craft_text_detector import Craft
from paddleocr import PaddleOCR

# from api_client import evaluate
from freshness_model import FreshnessModel
warnings.filterwarnings('ignore')
import datetime

import io
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()
GOOGLE_API_KEY = "AIzaSyC5mUr43dapOIIROSjIFWQMHRpQMpVGqfE"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
freshness_final_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# set to evaluation mode
freshness_final_model.eval()
with open("ban_model.pkl", "rb") as f:
    freshness_score_model = pickle.load(f)
print("Model loaded using pickle.")
f_model = load_model('C:/Users/yashg/Desktop/flipkart/ocr_backend/transfer_learningv4.h5') 
craft_detector = Craft(crop_type="box", cuda=False)
ocr = PaddleOCR(use_angle_cls=True, lang='en')
logging.getLogger("ppocr").setLevel(logging.CRITICAL)

# load COCO category names
COCO_CLASS_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'banana', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
def prepare_image(image_path):
  
    try:
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        byte_stream = io.BytesIO()
        img.save(byte_stream, format='JPEG')
        byte_stream.seek(0)
        return {
            "mime_type": "image/jpeg",
            "data": byte_stream.getvalue()
        }
    except Exception as e:
        print(f"Error preparing image: {e}")
        return None

def analyze_image(image_path, system_prompt):
    
    try:
        image_data = prepare_image(image_path)
        response = model.generate_content([system_prompt, image_data])
        return response.text
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

# Replace with the IP Webcam URL displayed on your Android app
# IP_WEBCAM_URL = "http://192.168.29.76:8080"
# PHOTO_SAVE_PATH = "captured_images"  # Directory to save images

# Load your custom YOLOv8 model weights
# yolo_model = YOLO("best.pt")  # Specify the path to your YOLO weights file
# brand_model = YOLO("best_weights_brand.pt") 
def predict_image(image_path, model, target_size=(150, 150)):
    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Rescale the image as per the training process
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)[0]  # Get the index of the highest probability
    
    return predicted_class, predictions[0]  # Return the predicted class index and probabilities

# Load the saved AlexNet model
alexnet = models.alexnet(pretrained=True)
alexnet.classifier[6] = nn.Linear(4096, 1)  # Modify the final layer
alexnet.load_state_dict(torch.load('alexnet_model.pth', map_location=torch.device('cpu')))
alexnet.eval()  # Set the model in evaluation mode

# Define the same transformations used during training for AlexNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 (AlexNet input size)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])



# result = CLIENT.infer(your_image.jpg, model_id="groceries-6pfog/6")

# Set the path to the Tesseract OCR executable
# pytesseract.pytesseract.tesseract_cmd = 'C:/Users/yashg/t_ocr/tesseract.exe'

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests

# @app.route("/")
# def home():
#     return render_template("index1.html")
# Load your custom YOLOv8 model weights
# counting_model = YOLO("best_weights.pt")  #weights for counting different items in image

# Load the saved AlexNet model
# alexnet = models.alexnet(pretrained=True)
# alexnet.classifier[6] = nn.Linear(4096, 1)  # Modify the final layer
# alexnet.load_state_dict(torch.load('alexnet_model.pth', map_location=torch.device('cpu')))
# alexnet.eval()  # Set the model in evaluation mode

# # Define the same transformations used during training for AlexNet
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to 224x224 (AlexNet input size)
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
# ])
def extract_dominant_hues(hue_values, k=5):
    """Extract dominant hues using k-means clustering emphasizing freshness stages."""
    hue_values = hue_values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(hue_values)
    return sorted(kmeans.cluster_centers_.flatten())[:3] 

def extract_features(img):
    resized_img = cv2.resize(img, (128, 128))

    # Extract dominant hues for color feature
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    hue = hsv_img[:, :, 0].flatten()
    dominant_hues = extract_dominant_hues(hue)

    # Texture Feature (GLCM using OpenCV)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    glcm = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    contrast = np.var(glcm)
    energy = np.sum(glcm**2)
    homogeneity = np.sum(glcm / (1 + np.abs(np.arange(256) - np.arange(256)[:, None])))

    texture_features = [contrast, energy, homogeneity]

    # Shape Feature (HOG)
    hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8), 
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

    # Combine features
    combined_features = np.hstack([dominant_hues, texture_features, hog_features])
    return combined_features
# os.environ['GOOGLE_API_KEY'] = 'AIzaSyCy64eHFj5NduOL0dPpIiMVdUvP8vcAFJY'
# genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
# gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def count_objects(image_path, resize_dim=(416, 416)):
    # Step 1: Read and Preprocess Image
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, resize_dim)
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Step 2: Edge Detection
    edges = cv2.Canny(blurred_image, 50, 200)

    # Step 3: Morphological Transformations
    kernel = np.ones((3, 3), np.uint8)
    morph_image = cv2.dilate(edges, kernel, iterations=1)

    # Display Morphological Image
    print("Morphological Image:")
    cv2.imshow(morph_image)

    # Step 4: Find Contours (Objects)
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw Contours (for visualization)
    output_image = resized_image.copy()
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    print("Objects Detected:")
    cv2.imshow(output_image)

    # Step 5: Count the Number of Contours
    object_count = len(contours)
    print(f"Total Number of Objects Detected: {object_count}")

    return object_count
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_model_dir='./models/det', rec_model_dir='./models/rec', cls_model_dir='./models/cls', use_gpu=False)

def days_between_dates(date1, date2):
    # Specify the date format
    date_format = "%d/%m/%y"
    
    # Convert strings to datetime objects
    d1 = datetime.datetime.strptime(str(date1), date_format)
    d2 = datetime.datetime.strptime(str(date2), date_format)
    
    # Calculate the difference in days
    delta = (d2 - d1).days
    return delta

def generate_answer(img_path):
    start=time.time()
    result = ocr.ocr(img_path)
    end=time.time()
    print(end-start)

    print("---------------------------")
    stri=""
    for i in range (0,len(result[0])):
        stri=stri+result[0][i][-1][0]+"\n"
    # print(stri)
    # print("----------------------------------------------")
    # text_data = "\n".join([item[-1][0] for item in result[0]])
    # prompt = f"""
    # Extract the following information from the text:

    # - brand_name: The name of the brand or company
    # - product_name: The specific name of the food item
    # - product_type: The general category of the food item (e.g., snack, beverage, dairy)
    # - ingredients: A list of ingredients with their quantities in grams
    # - **nutrition_facts: Nutritional information, including calories, fat, carbohydrates, protein, etc.
    # - manufacturing_date: The date the product was manufactured
    # - expiration_date: The expiration date of the product
    # - price: The price of the product

    # Provide the extracted information in a well-formatted text format.

    # Text:
    # {text_data}
    # """
    # final_str="Tell me company name of food item that makes it and food ingredient content with amount of gram present in it and also manufactoring date, price and expiry date from the text given as follows in a good format \""+stri+"\""
    now = datetime.datetime.now()
    final_str="Give fast reply in time.Fill this 0)Count of Objects in the image: \n1)Brand Name: \n2)Product Name:\n3) Manufacturing Date: \n4)Expiry Date: (Please give expiry date in \"%d/%m/%y\" format) \5)Net Quantity: \6)Price: \n7)Ingredient in grams:   Context is \""+stri+"\. There can be multiple objects details so give all of them seprately. In addition, every line doesnt mean related to line above and below so make thinking according to product's and brand name and give only answer with headings in normal text and not making it bold , dont add anything before and after. Also, if there are more than one brand item then list them seprately. Count total number of objects in the wh"
    start=time.time()
     # Initial analysis prompt
    initial_prompt = """
    Look at this image and determine:
    1. Is it a packaged product or fresh produce (fruit/vegetable)?
    2. How many items are visible in the image?
    3. Count slowly one at a time to ensure no errors
    4. See the full image 
    Respond in this format:
    Type: [packaged/fresh]
    Count: [number]
    """
    print("Analyzing the image...")
    initial_response = analyze_image(img_path, initial_prompt)
    if not initial_response:
        print("Failed to analyze the image.")
        return

    print("\nInitial Analysis:")
    print(initial_response)

    if "packaged" in initial_response.lower():
        # Packaged product prompt
        packaged_prompt = """
        For this packaged product:
        - Identify the brand name.
        - Find the expiry date (if visible).
        - Count the distinct physical items/packages.
        - Look for the MRP (Maximum Retail Price) if available.
        -if there are many brands in image only some are visible so write atleast visible brand   names instead of many brand names

        Respond in this format:
        Brand: [brand name]
        Product Name: [product or flavour if given in image]
        Expiry Date: [YYYY-MM-DD]
        Net Quantity: [in g]
        Ingredients: [all ingredients in g]
        MRP: [price in ₹ or Rs.]
        Count: [number]
        """
        details = analyze_image(img_path, packaged_prompt)
        print("\nPackaged Product Details:")
        return details
    # gemini_response = gemini_model.generate_content(final_str)
    # end=time.time()
    # print(gemini_response.text)
    # print(end-start)
    # return gemini_response.text
    
#     try:
#         gemini_response = gemini_model.generate_content(final_str)
#         print(gemini_response.text)
#         s1 = gemini_response.text
#         s2 = f"CurrentTimeStep :{now}"
#         result = str(s1) + " " + s2 + "\n"
#         expiry_date_match = re.search(r"Expiry Date: (\d{2}/\d{2}/\d{2})", result)
#         print(type(expiry_date_match))
#         if expiry_date_match is not None:
#             expiry_date_str = expiry_date_match.group(1)
#         # print(expiry_date_str,expiry_date_match)
#         current_date = datetime.datetime.now().strftime("%d/%m/%y")
#         # print(current_date,expiry_date_str)
#         if expiry_date_match is not None:
#             delta = days_between_dates(current_date,str(expiry_date_str))
#         else :
#             delta = None
#         if (delta is None):
#             s3 = f"LifeTime span :Not available"
#         elif (delta > 0): 
#             s3 = f"LifeTime span :{delta}"
#         else:
#             s3 = f"LifeTime span : Expired"
#         result = str(result) + " " + s3
#         end = time.time()
#         print(f"Time elapsed: {end - start:.2f} seconds")
#         return result

#     except Exception as e:
#         if isinstance(e, json.JSONDecodeError):
#             print(f"Error occurred: Invalid JSON data: {e}")
#         else:
#             print(f"Error occurred: {e}")
#         return str(e)



    
@app.route("/itemcount", methods=["POST"])
def item_count():
    try:
        # Retrieve the uploaded file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        
        image = request.files['image']
        image_path = './temp_image.png'
        image.save(image_path)
        
        # Validate the file type
        if not image.content_type.startswith('image/'):
            return jsonify({"error": "Invalid file type"}), 400

        
        client = Client("shouryap/ObjectCountingModel")
        start=time.time()
        result = client.predict(
                media_input=handle_file(image_path),
                text_input="Number of objects in the image. Give count only.",
                api_name="/ObjectCounting_Inference"
        )
        end=time.time()
        print(result)
        print(end-start)
        images_folder= r'C:\Users\yashg\Desktop\flipkart\ocr_backend\database\images'
        unique_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        image_path = os.path.join(images_folder, unique_filename)
        image.save(image_path)
        csv_file = r'C:\Users\yashg\Desktop\flipkart\ocr_backend\database\counting_results.csv'
        file_exists = os.path.isfile(csv_file)

        # Write result to the CSV file
        with open(csv_file, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if not file_exists:
                # Write header if file doesn't exist
                csv_writer.writerow(['image', 'count'])
            # Append the result
            csv_writer.writerow([unique_filename, result])

        # results = counting_model.predict(source=image_path, save=True)
        # detections = results[0].boxes
        # total_objects = len(detections)
        # result = CLIENT.infer(image_path, model_id="groceries-6pfog/6")
        # Count the number of detected objects
        # num_objects = len(result['predictions'])

        # Pass the temporary .jpg file to the inference method
        # result = CLIENT.infer(image_path, model_id="groceries-6pfog/6")

        # Clean up the temporary file
        # os.remove(image_path)

        # Ensure result is JSON serializable
        return jsonify({"result": result})
    except Exception as e:
        # Log the error traceback
        print("Error occurred:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500    

def get_coloured_mask(mask):
    """
    random_colour_masks
        parameters:
        - image - predicted masks
        method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img_path, confidence):
  """
  get_prediction
    parameters:
      - img_path - path of the input image
      - confidence - threshold to keep the prediction or not
    method:
      - Image is obtained from the image path
      - the image is converted to image tensor using PyTorch's Transforms
      - image is passed through the model to get the predictions
      - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
        ie: eg. segment of cat is made 1 and rest of the image is made 0

  """
  img = Image.open(img_path).convert('RGB')
  transform = T.Compose([T.ToTensor()])
  img = transform(img)
  pred = freshness_final_model([img])
  pred_score = list(pred[0]['scores'].detach().numpy())
  pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
  masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
  # print(pred[0]['labels'].numpy().max())
  pred_class = [COCO_CLASS_NAMES[i] for i in list(pred[0]['labels'].numpy())]
  pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
  masks = masks[:pred_t+1]
  pred_boxes = pred_boxes[:pred_t+1]
  pred_class = pred_class[:pred_t+1]
  return masks, pred_boxes, pred_class

def compute_iou(mask1, mask2):
    """
    Computes the IoU (Intersection over Union) between two binary masks.

    Args:
        mask1 (np.array): First binary mask (2D array).
        mask2 (np.array): Second binary mask (2D array).

    Returns:
        float: IoU value.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def segment_instance(img_path, confidence=0.5, rect_th=2, text_size=2, text_th=1):
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ratio 1:0.8 with opencv
        - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img_path, confidence)
    print(boxes)
    image_final = cv2.imread(img_path)
    bounding_boxes = [[(int(x1), int(y1)), (int(x2), int(y2))] for [(x1, y1), (x2, y2)] in boxes]
    idx = 0
    for (start_point, end_point) in bounding_boxes:
        color = (0, 255, 0)  # Green color
        thickness = 2  # Thickness of the rectangle border
        cv2.rectangle(image_final, start_point, end_point, color, thickness)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (0, 0, 255)  # Red color for text
        text_position = (start_point[0], start_point[1] - 10)  # Position above the bounding box
        cv2.putText(image_final, str(idx), text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        idx+=1
    print(pred_cls)
    cv2.imwrite("final.jpeg", image_final)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(masks)
    num_masks = len(masks)
    to_remove = set()
    for i in range(num_masks):
        for j in range(num_masks):
            if i != j:
                # Compute IoU between mask i and mask j
                iou = compute_iou(masks[i], masks[j])

                # If IoU exceeds threshold, remove the larger mask
                print(i,j,iou)
                if iou > 0.4:
                    # Determine which mask is larger by area
                    if masks[i].sum() > masks[j].sum():
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

    # Filter out the masks marked for removal
    filtered_masks = [mask for idx, mask in enumerate(masks) if idx not in to_remove]
    masks = filtered_masks
    final_images = []
    print(len(filtered_masks))
    for i in range(0,len(masks)):
      white_bg = np.ones_like(img) * 255
      img_with_mask = np.where(masks[i][:, :, None], img, white_bg)
      final_images.append(img_with_mask)
    #   plt.imshow(img_with_mask)
    #   plt.show()
    return final_images


#freshness
@app.route("/freshness", methods=["POST"])
# def fresh():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400
    
    
#     image = request.files['image']
#     image_path = './temp1_image.png'
#     image.save(image_path)
#      # if not image.content_type.startswith('image/'):
#     #     return jsonify({"error": "Invalid file type"}), 400
#     # image = Image.open(image_path).convert("RGB")

#     # results = yolo_model(image)  
#     # detections = results[0].boxes  
#     # for box in detections:
#     #     x1, y1, x2, y2 = box.xyxy[0].int().tolist()

#     #     cropped_img = image.crop((x1, y1, x2, y2))  
#     #     input_tensor = transform(cropped_img).unsqueeze(0)  
#     #     with torch.no_grad():
#     #         credibility_score = alexnet(input_tensor).item()

#     #     print(f"Detected object credibility score: {credibility_score:.2f}")
#     # return jsonify({"result": credibility_score})
    
#         # Fresh produce prompt
    



#     final_images = segment_instance(image_path, confidence=0.8)
#     details = []
#     for i in range(len(final_images)):
#         new_image = final_images[i]
#         features = extract_features(new_image).reshape(1, -1)
#         prediction = freshness_score_model.predict(features)
#         details.append(f'id: {i} and freshness-score : {prediction[0]}')

#     structured_details = []
#     for item in details:
#         id_part, freshness_part = item.split(' and ')
#         record = {
#             "id": id_part.split(': ')[1],
#             "freshness_score": freshness_part.split(': ')[1]
#         }
#         structured_details.append(record)

#     print(structured_details)


#     fresh_prompt = """
#     For this fresh produce:
#     - Identify the type of produce (e.g., banana, apple).
#     - Count the items visible in the image.
#     - Rate the freshness on a scale of 1-10 (1: very poor, 10: very fresh).
#     - Estimate shelf life in days based on visible conditions.
#     - There can be multiple objects so give the below format data for every food item. It is necessary so give all the details seprately and write data for each object in the corresponding block.
#     - Give me just the detail block and do not add anything in english at the beginning or end. I require just the data block given below and nothing else strictly!
#     -  Before this leave  then  Give Total count of objects a line also in the image seprately which is the final total count of objects in the image in a new line in format "Total Count of Objects: ".

#     Respond in this format:
#     Produce Type: [name]
#     Freshness Score: [score]
#     Shelf Life: [days] 
#     ,remove br 
#     then leave a line
#     """
#     # details = analyze_image(image_path, fresh_prompt)
#     print("\nFresh Produce Details:")
#     print(details)
#     return jsonify({"credibility_score": structured_details})

#     # if not image.content_type.startswith('image/'):
#     #     return jsonify({"error": "Invalid file type"}), 400
#     # image = Image.open(image_path).convert("RGB")

#     # results = yolo_model(image)  
#     # detections = results[0].boxes  
#     # for box in detections:
#     #     x1, y1, x2, y2 = box.xyxy[0].int().tolist()

#     #     cropped_img = image.crop((x1, y1, x2, y2))  
#     #     input_tensor = transform(cropped_img).unsqueeze(0)  
#     #     with torch.no_grad():
#     #         credibility_score = alexnet(input_tensor).item()

#     #     print(f"Detected object credibility score: {credibility_score:.2f}")
#     # return jsonify({"result": credibility_score})


def fresh():
    # model = FreshnessModel()
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    image_path = './temp_image.png'
    image.save(image_path)

    final_images = segment_instance(image_path, confidence=0.8)
    details = []
    idx = 0
    for i, temp_image in enumerate(final_images):
        try:
            # print(temp_image.shape)
            image_path = './temp1_image.png'
            rgb_array = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_array)
            img.save(image_path)
            predicted_class, probabilities = predict_image(image_path, f_model)

            class_labels={'freshapples': 0, 'freshbanana': 1, 'freshoranges': 2, 'rottenapples': 3, 'rottenbanana': 4, 'rottenoranges': 5}
            class_labels = {v: k for k, v in class_labels.items()}  # Invert the dictionary for easy lookup
            predicted_label = class_labels[predicted_class]
            # score = evaluate(temp_image)
            # score = model.predict(temp_image)
            max_val=-1
            max_val_idx=-1
            for i in range(0,6):
                if probabilities[i] > max_val:
                    max_val=probabilities[i]
                    max_val_idx=i

            second_pair_idx=-1

            if max_val_idx<3:
                second_pair_idx=max_val_idx+3
            else:
                second_pair_idx=max_val_idx-3
            # print(max_val_idx)
            # print(second_pair_idx)
            final_score=(-0.8*probabilities[min(max_val_idx,second_pair_idx)])+(0.2*(probabilities[max(max_val_idx,second_pair_idx)]))+0.8
            shelf_life = "0-2"
            if (final_score > 0 and final_score < 0.2):
                shelf_life = "5-6"
            elif (final_score < 0.4):
                shelf_life = "3-4"
            elif (final_score < 0.85):
                shelf_life = "2-3"
            if min(max_val_idx,second_pair_idx)==0:
                details.append(f'id: {idx} Freshness: {final_score} class: Apple,shelf_life : {shelf_life} ')
            elif min(max_val_idx,second_pair_idx)==1:
                details.append(f'id: {idx} Freshness: {final_score} class: Banana,shelf_life : {shelf_life}')
            else:
                details.append(f'id: {idx} Freshness: {final_score} class: Orange,shelf_life : {shelf_life}')


            # details.append(f'id: {idx} Freshness: {final_score} ')
            idx+=1
            print(idx)
        except ValueError as e:
            print(f"Error processing image {temp_image}: {e}")
            details.append(f"id: {idx} and freshness-score: Error")
    print(details)
    total_count = len(final_images)
    # response = f"Total Count of Objects: {total_count}\n\n"
    serve_results("final.jpeg")
    images_folder= r'C:\Users\yashg\Desktop\flipkart\ocr_backend\database\images'
    unique_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
    image_path1 = os.path.join(images_folder, unique_filename)
    image.save(image_path1)
    csv_file = r'C:\Users\yashg\Desktop\flipkart\ocr_backend\database\freshness_results.csv'
    file_exists = os.path.isfile(csv_file)

    # Write result to the CSV file
    with open(csv_file, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            # Write header if file doesn't exist
            csv_writer.writerow(['image', 'Freshness'])
        # Append the result
        csv_writer.writerow([unique_filename, details])
    # for item in structured_details:
    #     response += (
    #         f"Produce Type: Unknown\n"
    #         f"Freshness Score: {item['freshness_score']}\n"
    #         f"Shelf Life: {random.randint(1, 7)} days\n\n"
    #     )
    # print(response)
    # img = Image.open('final.jpeg')
    return jsonify({"result": details,"image_path" : 'final.jpeg'})

#brand
@app.route("/expirydate", methods=["POST"])
def expirydate():
    # YOLO_SCRIPT = "main_detection_yolov8.py"
    # YOLO_MODEL = "weights/Logo_Detection_Yolov8.pt"
    # UPLOAD_FOLDER = "uploads/"
    RESULT_FOLDER = "./results/"
    try:
        # Retrieve the uploaded file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        
        image = request.files['image']
        image_path = './brand_image.png'
        image.save(image_path)
        
        # Validate the file type
        if not image.content_type.startswith('image/'):
            return jsonify({"error": "Invalid file type"}), 400

        results = brand_model.predict(source=image_path, save=True)
        # if hasattr(results, 'plot'):
        #     prediction_image_path = os.path.join(RESULT_FOLDER, 'prediction_result.png')
        #     results[0].plot(save_path=prediction_image_path)
        detected_classes = []
        for result in results:
                detected_classes.extend([result.names[int(cls)] for cls in result.boxes.cls])
        print(detected_classes)
        
        # detections = results[0].boxes
        # total_objects = len(detections)
        # result = CLIENT.infer(image_path, model_id="groceries-6pfog/6")
        # Count the number of detected objects
        # num_object = len(result['predictions'])

        # Pass the temporary .jpg file to the inference method
        # result = CLIENT.infer(image_path, model_id="groceries-6pfog/6")

        # Clean up the temporary file
        # os.remove(image_path)

        # Ensure result is JSON serializable
        return jsonify({"result": detected_classes})
    except Exception as e:
        # Log the error traceback
        print("Error occurred:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500    
    # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # try:
    #     # Check if an image file is present in the request
    #     if 'image' not in request.files:
    #         return jsonify({"error": "No image file provided"}), 400
        
    #     image_file = request.files['image']
    #     print("here1")

    #     # Validate the file type
    #     if not image_file.content_type.startswith('image/'):
    #         return jsonify({"error": "Invalid file type"}), 400

    #     # Save the uploaded image with a unique name
    #     image_path=UPLOAD_FOLDER+"brand1.PNG"
    #     print("here2")
    #     # try:
    #     image_file.save(image_path)
    #     result_path='brand1_detected_logo'
    #     command = [
    #         "python",
    #         YOLO_SCRIPT,
    #         "--model", YOLO_MODEL,
    #         "--image", image_path,
    #         "--save-result"
    #     ]
    #     print("above subprocess")
    # #     # Execute the command and capture the output
    #     print("Command to run:", command)
    #     process = subprocess.run(command, capture_output=True, text=True)
    #     # print("Command executed")  # Log after the command runs
    #     # print("stdout:", process.stdout)  # Print standard output for debugging
    #     # print("stderr:", process.stderr)  # Print error output for debugging
    #     print("below subprocess")
    #     if process.returncode != 0:
    #         return jsonify({"error": "Detection failed", "details": process.stderr}), 500

    #     # Return the result (assume results are saved in a JSON format or similar)
    #     return jsonify({"message": "Detection successful", "result_path": result_path}), 200

    # except Exception as e:
    #     return jsonify({"error": f"An error occurred: {str(e)}"}), 500
    
# RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
RESULTS_FOLDER = os.getcwd()
@app.route('/runs/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_FOLDER,filename)

def is_contained(obj_box, text_box):
    """
    Check if a text box is contained within an object box.
    This is determined by checking if the center of the text box lies inside the object box.
    """
    x1_obj, y1_obj, x2_obj, y2_obj = obj_box
    x1_text, y1_text, x2_text, y2_text = text_box

    # Calculate the center of the text box
    center_x = (x1_text + x2_text) / 2
    center_y = (y1_text + y2_text) / 2

    # Check if the center is inside the object box
    return x1_obj <= center_x <= x2_obj and y1_obj <= center_y <= y2_obj


def iou(box1, box2):
    """Calculate Intersection Over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute the coordinates of the intersection rectangle
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    # Compute the area of the intersection rectangle
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2_p - x1_p) * (y2_p - y1_p)

    # Compute the union area
    union_area = area_box1 + area_box2 - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def filter_bounding_boxes(bounding_boxes, iou_threshold=0.3):
    filtered_boxes = []
    for box in bounding_boxes:
        keep = True
        for kept_box in filtered_boxes:
            if calculate_iou(box, kept_box) > iou_threshold:
                keep = False
                break
        if keep:
            filtered_boxes.append(box)
    return filtered_boxes


def detect_objects(image_path,client):
    """Detect objects using Gradio client."""
    result = client.predict(
        img=file(image_path),
        text_queries="food item packets, bottles or boxes",
        owl_threshold=0.16,
        api_name="/predict"
    )

    # Parse object bounding boxes
    bounding_boxes = []
    print(result[1])
    for line in result[1].split("\n"):
        if line:
            coords = list(map(int, line.split(", ")))
            if len(coords) == 4:
                bounding_boxes.append(tuple(coords))
    return bounding_boxes


def detect_text(image_path):
    """Detect text using combined CRAFT and PaddleOCR approach."""
    image = cv2.imread(image_path)
    craft_result = craft_detector.detect_text(image)
    text_regions = []

    for poly in craft_result["boxes"]:
        x_min, y_min = map(int, poly.min(axis=0))
        x_max, y_max = map(int, poly.max(axis=0))
        crop = image[y_min:y_max, x_min:x_max]
        ocr_result = ocr.ocr(crop, cls=True)
        
        if ocr_result[0]:
            text = " ".join([line[1][0] for line in ocr_result[0]])
            text_regions.append(((x_min, y_min, x_max, y_max), text))
            print(text)
    
    return text_regions

def combine_results(object_boxes, text_regions):
    """
    Assign text to objects based on containment.
    """
    results = {}

    for i, obj_box in enumerate(object_boxes, 1):
        results[f"Object {i}"] = []  # Initialize text for each object

        for text_box, text in text_regions:
            if is_contained(obj_box, text_box):
                results[f"Object {i}"].append(text)  # Assign text to the object

    return results

def visualize_combined_results(image_path, object_boxes, text_regions, results):
    """Visualize object and text boxes with combined results."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Draw object boxes
    for obj_box in object_boxes:
        draw.rectangle(obj_box, outline="red", width=5)

    # Draw text boxes and annotate with text
    for text_box, text in text_regions:
        draw.rectangle(text_box, outline="blue", width=2)
        draw.text((text_box[0], text_box[1] - 10), text, fill="blue")

    # Annotate object boxes with their associated texts
    for i, obj_box in enumerate(object_boxes, 1):
        text_list = results[f"Object {i}"]
        label = "\n".join(text_list)
        if label:
            draw.text((obj_box[0], obj_box[1] - 20), label, fill="green")

    output_path = image_path[:-4] + "_combined_output.png"
    image.save(output_path)
    print(f"Combined results visualized at: {output_path}")

def visualize_results(image_path, combined_results):
    """Visualize the combined results."""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for obj_label, obj_box, texts in combined_results:
        draw.rectangle(obj_box, outline="red", width=3)
        draw.text((obj_box[0], obj_box[1] - 10), obj_label, fill="red")

        for text in texts:
            draw.text((obj_box[0] + 5, obj_box[1] + 20), text, fill="blue")

    output_path = image_path[:-4] + "_combined_output.png"
    image.save(output_path)
    print(f"Combined results saved at: {output_path}")

# client_prduct_details = Client("shouryap/groundingdi")
client_product_details = Client("shouryap/ObjectCountingModel1")
@app.route("/productdetails", methods=["POST"])
def productdetails():
    try:
        # Retrieve the uploaded file
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files['image']

        # Validate the file type
        if not image_file.content_type.startswith('image/'):
            return jsonify({"error": "Invalid file type"}), 400

        # Save the image to a temporary location for processing
        # Save the image to a temporary location for processing
        img_path = './uploaded_image.jpg'
        # img_path = './ocr_1.jpg'
        # Save the image without resizing
        image_file.save(img_path)
        # Process the image and extract product info
        # extracted_info = generate_answer(img_path)
        # Initialize Gradio client
        # client = Client("shouryap/groundingdi")
        # object_boxes = detect_objects(img_path,client)
        start = time.time()
        # img_path = '/home/shourya/qwen/parle_test.PNG'

        # Predict bounding boxes from the model
        start=time.time()
        result = client_product_details.predict(
                media_input=handle_file(img_path),
                text_input='''if expiry date is not visible then use best before data to calculate the expiry date using manufacturing date(mfg) given,add the best before months to manufacturing date  .For this packaged product:
            - Identify the brand name.
            - Find the expiry date .
            - Count the distinct physical items/packages.
            - Look for the MRP (Maximum Retail Price) if available.
            -if there are many brands in image only some are visible so write atleast visible brand   names instead of many brand names

            Respond in this format:
            Brand: [brand name]
            Product Name: [product or flavour if given in image]
            Manufacturing Date: [YYYY-MM-DD or YYYY-MM]
            Expiry Date: [YYYY-MM-DD]
            Net Quantity: [in g]
            Ingredients: [all ingredients in g]
            MRP: [price in ₹ or Rs.]
            Count: [number]'''
        ,
                api_name="/ObjectCounting_Inference"
        )

        # end = time.time()
        # print(end - start)

        # Process the result to extract bounding boxes
        # a = result[1].split("\n")
        # unique = set()
        # for i in a:
        #     if i != '':
        #         unique.add(i)

        # Convert unique bounding boxes into a list of tuples
        # bounding_boxes = []
        # for item in unique:
        #     coords = item.split(", ")
        #     if len(coords) == 4:
        #         x1, y1, x2, y2 = map(int, coords)
        #         bounding_boxes.append((x1, y1, x2, y2))

        # Filter bounding boxes based on IoU
        # filtered_boxes = filter_bounding_boxes(bounding_boxes, iou_threshold=0.3)
        # print(filtered_boxes)
        # print(bounding_boxes)
        # text_regions = detect_text(img_path)
        # results = combine_results(bounding_boxes, text_regions)
        # print(results)
        # for obj, texts in results.items():
        #     print(f"{obj}: {texts}")
        # print(filtered_boxes)
        # print(text_regions)
        # print(results)
        # Process the image and extract product info
      

        # Return the extracted information
        # print(extracted_info)
        image = Image.open(img_path)
        images_folder= r'C:\Users\yashg\Desktop\flipkart\ocr_backend\database\images'
        unique_filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
        image_path1 = os.path.join(images_folder, unique_filename)
        image.save(image_path1)
        csv_file = r'C:\Users\yashg\Desktop\flipkart\ocr_backend\database\product_details_results.csv'
        file_exists = os.path.isfile(csv_file)

        # Write result to the CSV file
        with open(csv_file, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if not file_exists:
                # Write header if file doesn't exist
                csv_writer.writerow(['image', 'Product Details'])
            # Append the result
            csv_writer.writerow([unique_filename, result])
        return jsonify({"extracted_info": result})

    except Exception as e:
        # Log the error traceback
        print("Error occurred:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# def process_image1():
#     try:
#         # Retrieve the uploaded file
#         if 'image' not in request.files:
#             return jsonify({"error": "No image file provided"}), 400
        
        
#         image_file = request.files['image']
        
#         # Validate the file type
#         if not image_file.content_type.startswith('image/'):
#             return jsonify({"error": "Invalid file type"}), 400

#         # Process the image
#         image = Image.open(image_file)
#         text = pytesseract.image_to_string(image)

#         # Return the OCR result
#         return jsonify({"result": text})
#     except Exception as e:
#         # Log the error traceback
#         print("Error occurred:", traceback.format_exc())
#         return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
