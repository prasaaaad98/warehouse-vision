import requests
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
# import matplotlib.pyplot as plt

import logging
from gradio_client import Client, file
from PIL import Image, ImageDraw
import cv2
import numpy as np
from craft_text_detector import Craft
from paddleocr import PaddleOCR
import io
GOOGLE_API_KEY = "{insert key here}"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
system_prompt= """
#     For this fresh produce:
#     - Identify the type of produce (e.g., banana, apple).
#     - Rate the freshness on a scale of 1-10 (1: very poor, 10: very fresh).
#     - Estimate shelf life in days based on visible conditions.
#     - There can be multiple objects so give the below format data for every food item. It is necessary so give all the details seprately and write data for each object in the corresponding block.
#     - Give me just the detail block and do not add anything in english at the beginning or end. I require just the data block given below and nothing else strictly!

#     Respond in this format:
#     Produce Type: [name]
#     Freshness Score: [score]
#     Shelf Life: [days] 
#     ,remove br 
#     then leave a line
#     """
# Function to send an image to the API and get the result
def prepare_image(image_path):
    """
    Prepare image for Gemini API.
    """
    try:
        # print(image_path.shape)
        # img = Image.open(image_path)
        img = Image.fromarray(image_path)
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
def evaluate(image_path):
    """
    Analyze the image using Gemini API.
    """
    try:
        image_data = prepare_image(image_path)
        response = model.generate_content([system_prompt, image_data])
        return response.text
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None