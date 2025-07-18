from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import numpy as np
import cv2
import io
import os
from PIL import Image
from app.services.yolo_model_service import yolo_service
from app.services.cnn_classifier_service import cnn_classifier_service

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI in Docker!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/detect-chromosomes")
async def detect_chromosomes(file: UploadFile = File(...), classify: bool = False):
    """
    Upload an image and detect chromosomes using YOLO model
    
    Args:
        file: Image file to process
        classify: Whether to include CNN classification (optional, default False)
    
    Returns:
        JSON response with detection results including:
        - Number of detections
        - Bounding box coordinates
        - Confidence scores
        - Base64 encoded cropped images
        - Base64 encoded images with bounding boxes
        - CNN classification (if classify=True)
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the uploaded file
        contents = await file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert PIL Image to OpenCV format (numpy array)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array (RGB) then to BGR for OpenCV
        image_array = np.array(pil_image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Process the image with YOLO (and optionally CNN classification)
        if classify:
            results = yolo_service.process_image_with_classification_and_base64(image_array)
        else:
            results = yolo_service.process_image_with_base64_output(image_array)
        
        # Format response
        response = {
            "success": True,
            "filename": file.filename,
            "session_id": results["session_id"],
            "total_detections": results["total_detections"],
            "detections": [],
            "all_detections_image": results.get("all_detections_image_base64"),
            "all_detections_download_url": f"/download/{results['session_id']}/all-detections"
        }
        
        # Add classification summary if classification was performed
        if classify and results["detections"] and "classification_summary" in results["detections"][0]:
            response["classification_summary"] = results["detections"][0]["classification_summary"]
        
        # Format detection results for API response
        for detection in results["detections"]:
            cnn_result = detection.get("cnn_classification", {}) if classify else {}
            
            # Use CNN classification as the main class name if available and classification was requested
            if classify and cnn_result.get("classification"):
                main_class_name = cnn_result["classification"]
                yolo_class_name = detection["class_name"]
            else:
                main_class_name = detection["class_name"]
                yolo_class_name = detection["class_name"]
            
            formatted_detection = {
                "id": detection["id"],
                "class_name": main_class_name,
                "confidence": detection["confidence"],
                "bounding_box": {
                    "x1": detection["bbox"][0],
                    "y1": detection["bbox"][1], 
                    "x2": detection["bbox"][2],
                    "y2": detection["bbox"][3]
                },
                "cropped_image": detection.get("cropped_image_base64"),
                "bbox_image": detection.get("bbox_image_base64")
            }
            
            # Add CNN classification if it was performed
            if classify and "cnn_classification" in detection:
                formatted_detection["cnn_classification"] = detection["cnn_classification"]
                if cnn_result.get("classification"):
                    formatted_detection["yolo_class_name"] = yolo_class_name
            
            response["detections"].append(formatted_detection)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/model-info")
def get_model_info():
    """Get information about the loaded YOLO model"""
    try:
        # Get model information
        model_info = {
            "model_type": "YOLO",
            "model_path": "model_weights/yolo_model_weights.pt",
            "classes": yolo_service.yolo_model.names if hasattr(yolo_service.yolo_model, 'names') else {},
            "input_size": getattr(yolo_service.yolo_model, 'imgsz', 640)
        }
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/cnn-model-info")
def get_cnn_model_info():
    """Get information about the loaded CNN classification model"""
    try:
        return cnn_classifier_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CNN model info: {str(e)}")

@app.get("/all-models-info")
def get_all_models_info():
    """Get information about all loaded models"""
    try:
        yolo_info = {
            "model_type": "YOLO",
            "model_path": "model_weights/yolo_model_weights.pt",
            "classes": yolo_service.yolo_model.names if hasattr(yolo_service.yolo_model, 'names') else {},
            "input_size": getattr(yolo_service.yolo_model, 'imgsz', 640)
        }
        
        cnn_info = cnn_classifier_service.get_model_info()
        
        return {
            "yolo_model": yolo_info,
            "cnn_model": cnn_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting models info: {str(e)}")

@app.get("/download/{session_id}/{image_type}/{detection_id}")
async def download_image(session_id: str, image_type: str, detection_id: int):
    """
    Download processed images from a detection session
    
    Args:
        session_id: The session ID from the detection response
        image_type: Either 'cropped' or 'bbox' for the type of image
        detection_id: The ID of the specific detection (0, 1, 2, etc.)
    
    Returns:
        The requested image file for download
    """
    try:
        # Validate image_type
        if image_type not in ['cropped', 'bbox']:
            raise HTTPException(status_code=400, detail="image_type must be 'cropped' or 'bbox'")
        
        # Construct the file path based on the session and parameters
        base_output_folder = os.path.join(os.path.dirname(__file__), 'outputs')
        session_folder = os.path.join(base_output_folder, session_id)
        
        if image_type == 'cropped':
            image_folder = os.path.join(session_folder, 'cropped_predictions')
            # Find the file that matches the detection_id pattern
            files = [f for f in os.listdir(image_folder) if f.startswith(f'cropped_{detection_id}_')]
        else:  # bbox
            image_folder = os.path.join(session_folder, 'bbox_images')
            # Find the file that matches the detection_id pattern
            files = [f for f in os.listdir(image_folder) if f.startswith(f'bbox_{detection_id}_')]
        
        if not files:
            raise HTTPException(status_code=404, detail=f"No {image_type} image found for detection {detection_id} in session {session_id}")
        
        # Get the first matching file
        filename = files[0]
        file_path = os.path.join(image_folder, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # Return the file for download
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='image/jpeg'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading image: {str(e)}")

@app.get("/download/{session_id}/all/{image_type}")
async def download_all_images_info(session_id: str, image_type: str):
    """
    Get information about all available images for download in a session
    
    Args:
        session_id: The session ID from the detection response
        image_type: Either 'cropped' or 'bbox' for the type of images
    
    Returns:
        List of available images with download URLs
    """
    try:
        # Validate image_type
        if image_type not in ['cropped', 'bbox']:
            raise HTTPException(status_code=400, detail="image_type must be 'cropped' or 'bbox'")
        
        # Construct the folder path
        base_output_folder = os.path.join(os.path.dirname(__file__), 'outputs')
        session_folder = os.path.join(base_output_folder, session_id)
        
        if image_type == 'cropped':
            image_folder = os.path.join(session_folder, 'cropped_predictions')
        else:  # bbox
            image_folder = os.path.join(session_folder, 'bbox_images')
        
        if not os.path.exists(image_folder):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Get all image files
        files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        
        # Create download info for each file
        download_info = []
        for filename in files:
            # Extract detection ID from filename
            if image_type == 'cropped' and filename.startswith('cropped_'):
                detection_id = filename.split('_')[1]
            elif image_type == 'bbox' and filename.startswith('bbox_'):
                detection_id = filename.split('_')[1]
            else:
                continue
            
            download_info.append({
                "detection_id": int(detection_id),
                "filename": filename,
                "download_url": f"/download/{session_id}/{image_type}/{detection_id}"
            })
        
        return {
            "session_id": session_id,
            "image_type": image_type,
            "total_images": len(download_info),
            "images": download_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting download info: {str(e)}")

@app.get("/download/{session_id}/all-detections")
async def download_all_detections_image(session_id: str):
    """
    Download the image with all bounding boxes drawn on it
    
    Args:
        session_id: The session ID from the detection response
    
    Returns:
        The image file with all detections marked with bounding boxes
    """
    try:
        # Construct the file path
        base_output_folder = os.path.join(os.path.dirname(__file__), 'outputs')
        session_folder = os.path.join(base_output_folder, session_id)
        file_path = os.path.join(session_folder, 'all_detections.jpg')
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"All detections image not found for session {session_id}")
        
        # Return the file for download
        return FileResponse(
            path=file_path,
            filename=f"all_detections_{session_id}.jpg",
            media_type='image/jpeg'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading all detections image: {str(e)}")

@app.post("/classify-chromosomes")
async def classify_chromosomes(file: UploadFile = File(...)):
    """
    Upload an image and classify chromosomes using CNN classifier
    
    Returns:
        JSON response with classification results including:
        - Class label
        - Confidence score
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the uploaded file
        contents = await file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert PIL Image to OpenCV format (numpy array)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array (RGB) then to BGR for OpenCV
        image_array = np.array(pil_image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Classify the image with CNN
        results = cnn_classifier_service.classify_image(image_array)
        
        # Format response
        response = {
            "success": True,
            "filename": file.filename,
            "class_label": results["class_label"],
            "confidence": results["confidence"]
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-and-classify-chromosomes")
async def detect_and_classify_chromosomes(file: UploadFile = File(...)):
    """
    Upload an image, detect chromosomes using YOLO, and classify each detection using CNN
    
    Returns:
        JSON response with detection and classification results including:
        - Number of detections
        - Bounding box coordinates
        - Confidence scores
        - CNN classification (dicentric/not dicentric)
        - Classification confidence
        - Base64 encoded cropped images
        - Base64 encoded images with bounding boxes
        - Summary statistics
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the uploaded file
        contents = await file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert PIL Image to OpenCV format (numpy array)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to numpy array (RGB) then to BGR for OpenCV
        image_array = np.array(pil_image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Process the image with YOLO and CNN classification
        results = yolo_service.process_image_with_classification_and_base64(image_array)
        
        # Format response
        response = {
            "success": True,
            "filename": file.filename,
            "session_id": results["session_id"],
            "total_detections": results["total_detections"],
            "detections": [],
            "all_detections_image": results.get("all_detections_image_base64"),
            "all_detections_download_url": f"/download/{results['session_id']}/all-detections",
            "classification_summary": None
        }
        
        # Extract classification summary if available
        if results["detections"] and "classification_summary" in results["detections"][0]:
            response["classification_summary"] = results["detections"][0]["classification_summary"]
        
        # Format detection results for API response
        for detection in results["detections"]:
            cnn_result = detection.get("cnn_classification", {})
            
            # Use CNN classification as the main class name if available
            if cnn_result.get("classification"):
                main_class_name = cnn_result["classification"]
                yolo_class_name = detection["class_name"]
            else:
                main_class_name = detection["class_name"]
                yolo_class_name = detection["class_name"]
            
            formatted_detection = {
                "id": detection["id"],
                "class_name": main_class_name,  # CNN classification result
                "yolo_class_name": yolo_class_name,  # Original YOLO detection class
                "confidence": detection["confidence"],
                "bounding_box": {
                    "x1": detection["bbox"][0],
                    "y1": detection["bbox"][1], 
                    "x2": detection["bbox"][2],
                    "y2": detection["bbox"][3]
                },
                "cropped_image": detection.get("cropped_image_base64"),
                "bbox_image": detection.get("bbox_image_base64"),
                "cnn_classification": cnn_result
            }
            response["detections"].append(formatted_detection)
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/classify-detection/{session_id}/{detection_id}")
async def classify_single_detection(session_id: str, detection_id: int):
    """
    Classify a single detection from a previous session using CNN
    
    Args:
        session_id: The session ID from a previous detection
        detection_id: The ID of the specific detection to classify
    
    Returns:
        Classification results for the specified detection
    """
    try:
        # Construct the path to the cropped image
        base_output_folder = os.path.join(os.path.dirname(__file__), 'outputs')
        session_folder = os.path.join(base_output_folder, session_id)
        cropped_folder = os.path.join(session_folder, 'cropped_predictions')
        
        if not os.path.exists(cropped_folder):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Find the cropped image file for this detection
        files = [f for f in os.listdir(cropped_folder) if f.startswith(f'cropped_{detection_id}_')]
        
        if not files:
            raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found in session {session_id}")
        
        # Use the first matching file
        image_path = os.path.join(cropped_folder, files[0])
        
        # Classify the detection
        classification_result = cnn_classifier_service.classify_detection_from_path(image_path)
        
        return {
            "session_id": session_id,
            "detection_id": detection_id,
            "filename": files[0],
            "classification": classification_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying detection: {str(e)}")