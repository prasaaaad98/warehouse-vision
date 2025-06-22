from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2 # Still imported, but less used directly for Gemini calls
import numpy as np # Still imported, but less used directly for Gemini calls
from PIL import Image # Used for getting image dimensions if needed
import io
import json
from typing import List, Dict, Any, Optional
import os
import httpx # For making API calls to Gemini
from dotenv import load_dotenv
import datetime
from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta
import re 
# Load environment variables (for GEMINI_API_KEY)
load_dotenv()

app = FastAPI(title="Warehouse Vision API", version="1.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in .env file. Gemini calls will fail.")
    # raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# --- Gemini API Helper ---
async def call_gemini_api(
    image_bytes: bytes, 
    mime_type: str, 
    prompt: str, 
    output_schema: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Calls the Gemini API with an image and a prompt, optionally expecting structured JSON output.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key is not configured on the server.")

    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": mime_type, "data": image_bytes.decode('latin-1')}} # Send bytes as base64 later
    ]
    
    # Convert image_bytes to actual base64 for inline_data
    import base64
    base64_image_data = base64.b64encode(image_bytes).decode('utf-8')
    parts[1] = {"inline_data": {"mime_type": mime_type, "data": base64_image_data}}

    payload: Dict[str, Any] = {"contents": [{"role": "user", "parts": parts}]}

    if output_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": output_schema,
        }

    async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout
        try:
            print(f"Calling Gemini API. Prompt: {prompt[:100]}... Schema: {'Yes' if output_schema else 'No'}")
            response = await client.post(GEMINI_API_URL, json=payload)
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses
            
            result = response.json()
            print("Gemini API raw response:", json.dumps(result, indent=2)[:500]) # Log part of the response

            if not result.get("candidates") or not result["candidates"][0].get("content") or not result["candidates"][0]["content"].get("parts"):
                print("Error: Unexpected Gemini API response structure", result)
                raise HTTPException(status_code=500, detail="Unexpected response structure from Gemini API.")

            # If schema was used, the response part should be JSON text
            api_response_text = result["candidates"][0]["content"]["parts"][0]["text"]
            if output_schema:
                return json.loads(api_response_text) # Parse the JSON string
            return api_response_text # Return raw text if no schema

        except httpx.HTTPStatusError as e:
            print(f"Gemini API HTTPStatusError: {e.response.status_code} - {e.response.text}")
            error_detail = f"Gemini API request failed with status {e.response.status_code}."
            try:
                error_body = e.response.json()
                error_detail += f" Message: {error_body.get('error', {}).get('message', 'Unknown error')}"
            except json.JSONDecodeError:
                error_detail += f" Response: {e.response.text}"
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except httpx.RequestError as e:
            print(f"Gemini API RequestError: {e}")
            raise HTTPException(status_code=503, detail=f"Could not connect to Gemini API: {str(e)}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Gemini response: {e}. Response text: {api_response_text}")
            raise HTTPException(status_code=500, detail="Error decoding JSON response from Gemini API.")
        except Exception as e:
            print(f"An unexpected error occurred while calling Gemini API: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# --- Updated Service Functions (using Gemini) ---

async def brand_recognition(image_bytes: bytes, mime_type: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    prompt = f"""
    Analyze this image (dimensions: {image_width}x{image_height} pixels) to identify product brands and their specific products.
    For each distinct product variant found, provide:
    1. 'brand_name': The primary brand (e.g., "Parle", "Lays", "Coca-Cola").
    2. 'product_name': The specific product name under that brand (e.g., "Parle-G", "Magic Masala", "Coke Zero").
    3. 'variant_description': Any distinguishing features of this specific product variant if visible (e.g., "red packet", "500g size", "Classic Salted flavor", "blue label"). If no specific variant is noted or it's the standard version, return "Standard" or "N/A".
    4. 'quantity': An integer representing the count of this exact brand/product/variant combination visible in the image.
    5. 'confidence': A float score (0.0 to 1.0) for the overall identification of this entry.

    Example: If you see three "Parle-G 50g red packets" and two "Parle Hide & Seek Caffe Mocha packets", you should return two separate entries in the list.
    If no brands or products are clearly identifiable, return an empty list.
    Focus on distinct product offerings. Do not provide bounding boxes.
    """
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "brand_name": {
                    "type": "STRING",
                    "description": "The primary brand name (e.g., 'Parle', 'Lays')."
                },
                "product_name": {
                    "type": "STRING",
                    "description": "The specific product name (e.g., 'Parle-G', 'Magic Masala')."
                },
                "variant_description": {
                    "type": "STRING",
                    "description": "Description of the specific variant if distinguishable (e.g., 'red packet', '500g', 'Classic Salted', or 'N/A')."
                },
                "quantity": {
                    "type": "INTEGER",
                    "description": "The count of this specific brand/product/variant visible in the image."
                },
                "confidence": {
                    "type": "NUMBER",
                    "description": "Confidence score for this entire entry (0.0 to 1.0)."
                }
            },
            "required": [
                "brand_name",
                "product_name",
                "variant_description",
                "quantity",
                "confidence"
            ]
        }
    }
    try:
        response_data = await call_gemini_api(image_bytes, mime_type, prompt, schema)
        # Ensure it's a list, and default fields if somehow missing despite schema
        if isinstance(response_data, list):
            for item in response_data:
                item.setdefault("brand_name", "Unknown Brand")
                item.setdefault("product_name", "Unknown Product")
                item.setdefault("variant_description", "N/A")
                item.setdefault("quantity", 1) # Default to 1 if not specified
                item.setdefault("confidence", 0.5) # Default confidence
            return response_data
        return [] 
    except Exception as e:
        print(f"Error in gemini_brand_recognition processing: {e}")
        import traceback
        traceback.print_exc()
        return []


async def freshness_detection(image_bytes: bytes, mime_type: str) -> List[Dict[str, Any]]:
    prompt = """
    
    Analyze this image of produce. Identify all the fruits items.
    For each item, provide its type (e.g., 'apple', 'banana', 'lettuce'),
    a freshness score (a float between 0.0 for very poor and 1.0 for very fresh),
    a freshness level (string: 'Fresh', 'Moderate', 'Poor'),
    and a list of 2-3 brief recommendations for storage or use based on its freshness.
    If no produce is clearly identifiable, return an empty list.
    """
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "item_type": {"type": "STRING", "description": "Type of the produce item."},
                "freshness_score": {"type": "NUMBER", "description": "Freshness score (0.0 to 1.0)."},
                "freshness_level": {"type": "STRING", "enum": ["Fresh", "Moderate", "Poor"], "description": "Categorical freshness level."},
                "recommendations": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"},
                    "description": "Storage or use recommendations."
                }
            },
            "required": ["item_type", "freshness_score", "freshness_level", "recommendations"]
        }
    }
    try:
        return await call_gemini_api(image_bytes, mime_type, prompt, schema)
    except Exception as e:
        print(f"Error in gemini_freshness_detection processing: {e}")
        return []

async def item_counting(image_bytes: bytes, mime_type: str, image_width: int, image_height: int) -> Dict[str, Any]:
    prompt = f"""
    Analyze this warehouse image (dimensions: {image_width}x{image_height} pixels).
    Identify different types of items (e.g., 'box', 'bottle', 'pallet', 'can', 'package').
    For each distinct item type found, provide:
    1. The 'item_type' (string).
    2. The 'count' (integer) of how many such items are visible.
    3. A 'confidence' score (float, 0.8 to 0.99) for the identification of this item type.
    4. A list of 'locations', where each location is a plausible mock bounding box [x1, y1, x2, y2] for each counted instance.
       Ensure coordinates are integers within image bounds.
    Finally, provide a 'total_count' which is the sum of counts of all item types.
    If no items are clearly identifiable, return an empty list for 'items' and 0 for 'total_count'.
    """
    schema = {
        "type": "OBJECT",
        "properties": {
            "items": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "item_type": {"type": "STRING"},
                        "count": {"type": "INTEGER"},
                        "confidence": {"type": "NUMBER"},
                        "locations": {
                            "type": "ARRAY",
                            "items": {
                                "type": "ARRAY",
                                "items": {"type": "INTEGER"},
                                "minItems": 4,
                                "maxItems": 4
                            }
                        }
                    },
                    "required": ["item_type", "count", "confidence", "locations"]
                }
            },
            "total_count": {"type": "INTEGER"}
        },
        "required": ["items", "total_count"]
    }
    try:
        response_data = await call_gemini_api(image_bytes, mime_type, prompt, schema)
        # Basic validation and bounding box fix
        if isinstance(response_data, dict) and "items" in response_data:
            for item in response_data["items"]:
                if "locations" in item and isinstance(item["locations"], list):
                    for i, loc in enumerate(item["locations"]):
                        if isinstance(loc, list) and len(loc) == 4:
                            item["locations"][i] = [
                                max(0, min(loc[0], image_width)), 
                                max(0, min(loc[1], image_height)),
                                max(0, min(loc[2], image_width)),
                                max(0, min(loc[3], image_height))
                            ]
                            if item["locations"][i][0] >= item["locations"][i][2]: item["locations"][i][2] = item["locations"][i][0] + 10
                            if item["locations"][i][1] >= item["locations"][i][3]: item["locations"][i][3] = item["locations"][i][1] + 10
            return response_data
        return {"items": [], "total_count": 0} # Fallback
    except Exception as e:
        print(f"Error in gemini_item_counting processing: {e}")
        return {"items": [], "total_count": 0}


async def nutrition_extraction(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    prompt = """
    Analyze this image, assuming it is a nutrition label from a food product.
    Extract the following information if available:
    - 'product_name' (string, or "Unknown" if not found)
    - 'serving_size' (string, e.g., "1 cup (240ml)" or "50g". If not found, return an empty string or "N/A")
    - 'calories' (integer. If not found, return 0 or a similar placeholder integer if the schema requires an integer.)
    - 'nutrients': A list of objects. Each object should represent a single nutrient fact and have the following properties:
        - 'nutrient_name' (string, e.g., "Total Fat", "Sodium", "Protein")
        - 'amount' (string, e.g., "10g")
        - 'daily_value' (string, e.g., "15% DV", or an empty string "" if not applicable/found).
      Extract as many such nutrient facts as are clearly visible. If no nutrients are found, this list should be empty [].
      Prioritize common nutrients like: Total Fat, Saturated Fat, Trans Fat, Cholesterol, Sodium, Total Carbohydrate, Dietary Fiber, Total Sugars, Added Sugars, Protein, Vitamin D, Calcium, Iron, Potassium.
    - 'ingredients' (a list of strings, or an empty list [] if not found).
    - 'allergens' (a list of strings, e.g., ["Contains wheat", "May contain nuts"], or an empty list [] if not found).
    Ensure the entire output is valid JSON. If the image is not a nutrition label, return an object with appropriate placeholder values for all fields as per their types.
    """
    schema = {
        "type": "OBJECT",
        "properties": {
            "product_name": {"type": "STRING", "description": "The name of the product."},
            "serving_size": {"type": "STRING", "description": "Serving size information."},
            "calories": {"type": "INTEGER", "description": "Calories per serving."},
            "nutrients": { # Changed to ARRAY of objects
                "type": "ARRAY",
                "description": "List of identified nutrients and their values.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "nutrient_name": {"type": "STRING", "description": "Name of the nutrient (e.g., Total Fat, Sodium)."},
                        "amount": {"type": "STRING", "description": "Amount of the nutrient (e.g., 10g)."},
                        "daily_value": {"type": "STRING", "description": "Daily Value percentage (e.g., 15% DV), can be empty."}
                    },
                    "required": ["nutrient_name", "amount", "daily_value"] # daily_value can be empty string
                }
            },
            "ingredients": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List of ingredients."},
            "allergens": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List of allergens."}
        },
        "required": ["product_name", "serving_size", "calories", "nutrients", "ingredients", "allergens"]
    }
    try:
        data = await call_gemini_api(image_bytes, mime_type, prompt, schema)
        # Ensure the basic structure and default values are present
        data.setdefault("product_name", "Unknown")
        data.setdefault("serving_size", "N/A") 
        data.setdefault("calories", 0) 
        data.setdefault("nutrients", []) # Crucial default: empty list for nutrients
        data.setdefault("ingredients", [])
        data.setdefault("allergens", [])
        
        # Further ensure that if nutrients is present, it's a list
        if not isinstance(data.get("nutrients"), list):
            data["nutrients"] = []
            
        return data
    except Exception as e:
        print(f"Error in gemini_nutrition_extraction processing: {e}")
        return {
            "product_name": "Error processing label", "serving_size": "N/A", "calories": 0,
            "nutrients": [{"nutrient_name": "Error", "amount": str(e), "daily_value":""}], 
            "ingredients": [], "allergens": []
        }
async def mrp_expiry_detection(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    prompt = """
    Analyze this image for product packaging information. Extract the following details if present:
    - 'mrp_text': The Maximum Retail Price (MRP) as a string (e.g., "₹150.00", "USD 12.99"). If not found, return "N/A".
    - 'manufacturing_date_text': The manufacturing date (MFD) as a string (e.g., "01/2024", "Jan 2024", "MFD: 23 JAN 2024"). If not found, return "N/A".
    - 'best_before_text': The 'best before' period or date as a string (e.g., "Best before 12 months from MFD", "Use by 180 days", "Best Before: Oct 2025"). If not found, return "N/A".
    - 'expiry_date_text': Any explicitly printed expiry date (EXP) as a string (e.g., "EXP 12/2025", "Expiry: Jan 25", "Use By: 23/01/2026"). If not found, return "N/A".
    Return all fields, using "N/A" if a specific piece of information is not found. Ensure the output is valid JSON.
    """
    schema = {
        "type": "OBJECT",
        "properties": {
            "mrp_text": {"type": "STRING", "description": "Maximum Retail Price as text."},
            "manufacturing_date_text": {"type": "STRING", "description": "Manufacturing date as text."},
            "best_before_text": {"type": "STRING", "description": "Best before information as text."},
            "expiry_date_text": {"type": "STRING", "description": "Explicit expiry date as text."}
        },
        "required": ["mrp_text", "manufacturing_date_text", "best_before_text", "expiry_date_text"]
    }
    
    # --- Date parsing and calculation logic ---
    current_date = datetime.date.today()

    def parse_date_flexible(date_str: str, dayfirst_preference: bool = False) -> Optional[datetime.date]:
        if not date_str or date_str.lower() == "n/a" or date_str.lower() == "unknown":
            return None
        
        # Try specific DD/MM/YYYY or DD-MM-YYYY formats first if dayfirst_preference is True
        if dayfirst_preference:
            try:
                if "/" in date_str:
                    return datetime.datetime.strptime(date_str, "%d/%m/%Y").date()
                elif "-" in date_str:
                     return datetime.datetime.strptime(date_str, "%d-%m-%Y").date()
            except ValueError:
                pass # If specific format fails, fall through to dateutil.parser

        # General parsing with dateutil.parser
        try:
            # Pass dayfirst preference to dateutil.parser as well
            # Remove fuzzy=True for stricter initial parsing, can add as fallback if needed
            return dateutil_parser.parse(date_str, dayfirst=dayfirst_preference).date()
        except (ValueError, TypeError, OverflowError, dateutil_parser.ParserError):
            # Fallback for month/year formats
            try:
                if re.match(r"(\d{1,2})[/-](\d{4})", date_str): # MM/YYYY or MM-YYYY
                     # Assuming MM/YYYY here as it's a common fallback
                     dt_obj = datetime.datetime.strptime(date_str, "%m/%Y") if "/" in date_str else datetime.datetime.strptime(date_str, "%m-%Y")
                     return datetime.date(dt_obj.year, dt_obj.month, 1) # Default to 1st of the month
                if re.match(r"([a-zA-Z]{3,})\s+(\d{4})", date_str, re.IGNORECASE): # Mon YYYY
                     dt_obj = dateutil_parser.parse(date_str, default=datetime.datetime(1,1,1)).date() 
                     return datetime.date(dt_obj.year, dt_obj.month, 1) # Default to 1st of the month
            except: # Broad except for these specific fallback patterns
                pass
            print(f"Warning: Could not parse date string reliably: {date_str}")
            return None

    def calculate_shelf_life_status(expiry_dt: Optional[datetime.date]) -> str:
        if not expiry_dt:
            return "N/A"
        
        delta = relativedelta(expiry_dt, current_date)
        
        if expiry_dt < current_date:
            ago_delta = relativedelta(current_date, expiry_dt)
            if ago_delta.years > 0: return f"Expired {ago_delta.years} year{'s' if ago_delta.years > 1 else ''} ago"
            if ago_delta.months > 0: return f"Expired {ago_delta.months} month{'s' if ago_delta.months > 1 else ''} ago"
            return f"Expired {ago_delta.days} day{'s' if ago_delta.days > 1 else ''} ago"
        elif expiry_dt == current_date:
            return "Expires today"
        else: 
            parts = []
            if delta.years > 0: parts.append(f"{delta.years} year{'s' if delta.years > 1 else ''}")
            if delta.months > 0: parts.append(f"{delta.months} month{'s' if delta.months > 1 else ''}")
            if delta.days > 0 and delta.years == 0 : 
                 parts.append(f"{delta.days} day{'s' if delta.days > 1 else ''}")
            if not parts: return f"Expires on {expiry_dt.strftime('%Y-%m-%d')}" 
            return f"Expires in {', '.join(parts)}"

    try:
        extracted_data = await call_gemini_api(image_bytes, mime_type, prompt, schema)
        
        extracted_data.setdefault("mrp_text", "N/A")
        extracted_data.setdefault("manufacturing_date_text", "N/A")
        extracted_data.setdefault("best_before_text", "N/A")
        extracted_data.setdefault("expiry_date_text", "N/A")

        # Explicitly try parsing as DD/MM/YYYY first for these fields
        mfg_date = parse_date_flexible(extracted_data.get("manufacturing_date_text"), dayfirst_preference=True)
        exp_date_on_label = parse_date_flexible(extracted_data.get("expiry_date_text"), dayfirst_preference=True)
        
        calculated_exp_date: Optional[datetime.date] = None
        
        if exp_date_on_label:
            calculated_exp_date = exp_date_on_label
            print(f"Used explicit expiry date from label: {exp_date_on_label}")
        elif mfg_date and extracted_data.get("best_before_text", "N/A").lower() != "n/a":
            bb_text = extracted_data["best_before_text"].lower()
            delta_args = {}
            
            years_match = re.search(r"(\d+)\s*(?:year|yr)", bb_text)
            months_match = re.search(r"(\d+)\s*month", bb_text)
            days_match = re.search(r"(\d+)\s*day", bb_text)

            if years_match: delta_args["years"] = int(years_match.group(1))
            if months_match: delta_args["months"] = int(months_match.group(1))
            if days_match: delta_args["days"] = int(days_match.group(1))
            
            if delta_args:
                calculated_exp_date = mfg_date + relativedelta(**delta_args)
                print(f"Calculated expiry from MFD ({mfg_date}) + Best Before ({bb_text}): {calculated_exp_date}")
            else:
                exp_date_from_bb_text = parse_date_flexible(extracted_data.get("best_before_text"), dayfirst_preference=True)
                if exp_date_from_bb_text:
                    calculated_exp_date = exp_date_from_bb_text
                    print(f"Used best_before_text as an explicit date: {calculated_exp_date}")
        
        shelf_status = calculate_shelf_life_status(calculated_exp_date)

        return {
            "mrp": extracted_data.get("mrp_text", "N/A"),
            "manufacturing_date_str": extracted_data.get("manufacturing_date_text", "N/A"),
            "best_before_str": extracted_data.get("best_before_text", "N/A"),
            "expiry_date_on_label_str": extracted_data.get("expiry_date_text", "N/A"),
            "parsed_mfg_date": mfg_date.strftime("%Y-%m-%d") if mfg_date else "N/A",
            "parsed_expiry_date_on_label": exp_date_on_label.strftime("%Y-%m-%d") if exp_date_on_label else "N/A",
            "calculated_expiry_date": calculated_exp_date.strftime("%Y-%m-%d") if calculated_exp_date else "N/A",
            "shelf_life_status": shelf_status
        }

    except Exception as e:
        print(f"Error in gemini_mrp_expiry_detection processing: {e}")
        import traceback
        traceback.print_exc()
        return {
            "mrp": "Error", "manufacturing_date_str": "Error", "best_before_str": "Error", 
            "expiry_date_on_label_str": "Error", "parsed_mfg_date": "Error", 
            "parsed_expiry_date_on_label": "Error", "calculated_expiry_date": "Error", 
            "shelf_life_status": f"Error processing: {str(e)}"
        }


# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Warehouse Vision  is running!"}

async def get_image_dims(image_bytes: bytes) -> (int, int):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return img.width, img.height
    except Exception:
        return 640, 480 # Default fallback


@app.post("/api/brand-recognition")
async def brand_recognition_endpoint(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await image.read()
    width, height = await get_image_dims(image_bytes)
    brands_data = await brand_recognition(image_bytes, image.content_type, width, height)
    return JSONResponse(content={
        "success": True,
        "brands": brands_data,
        "message": f"Processed brand recognition . Found {len(brands_data)} brand(s)."
    })

@app.post("/api/freshness-detection")
async def freshness_detection_endpoint(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await image.read()
    items_data = await freshness_detection(image_bytes, image.content_type)
    return JSONResponse(content={
        "success": True,
        "items": items_data,
        "message": f"Processed freshness detection . Analyzed {len(items_data)} item(s)."
    })

@app.post("/api/item-counting")
async def item_counting_endpoint(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await image.read()
    width, height = await get_image_dims(image_bytes)
    counting_result = await item_counting(image_bytes, image.content_type, width, height)
    return JSONResponse(content={
        "success": True,
        **counting_result, # Includes 'items' and 'total_count'
        "message": f"Processed item counting. Counted {counting_result.get('total_count', 0)} total items."
    })

@app.post("/api/nutrition-info")
async def nutrition_info_endpoint(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await image.read()
    nutrition_data = await nutrition_extraction(image_bytes, image.content_type)
    return JSONResponse(content={
        "success": True,
        "nutrition_info": nutrition_data,
        "message": "Processed nutrition information extraction using Gemini."
    })
@app.post("/api/mrp-expiry")
async def mrp_expiry_detection_endpoint(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    image_bytes = await image.read()
    data = await mrp_expiry_detection(image_bytes, image.content_type)
    return JSONResponse(content={
        "success": True,
        "mrp_expiry_info": data, # Consistent key for the data object
        "message": "Processed MRP & Expiry Date detection."
    })

if __name__ == "__main__":
    print(f"Starting Uvicorn server.  is {'SET' if GEMINI_API_KEY else 'NOT SET'}.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
