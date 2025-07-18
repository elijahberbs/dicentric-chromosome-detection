from ultralytics import YOLO
import os
import cv2
import shutil
import numpy as np
import base64
from typing import List, Dict, Any
from io import BytesIO
import uuid
from .cnn_classifier_service import cnn_classifier_service

class YOLOModelService:
    def __init__(self):
        # Load the YOLO model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model_weights', 'yolo_model_weights.pt')
        self.yolo_model = YOLO(model_path)
        
        # Base output folders
        self.base_output_folder = os.path.join(os.path.dirname(__file__), '..', 'outputs')
        self.ensure_output_directories()
    
    def ensure_output_directories(self):
        """Ensure output directories exist"""
        os.makedirs(self.base_output_folder, exist_ok=True)
    
    def delete_directory(self, directory: str):
        """Function to delete directories if they exist"""
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Deleted existing directory: {directory}")
        else:
            print(f"Directory does not exist: {directory}")
    
    def process_image(self, image_array: np.ndarray, session_id: str = None) -> Dict[str, Any]:
        """
        Process an image using YOLO model and return detection results
        
        Args:
            image_array: Input image as numpy array
            session_id: Optional session ID for organizing outputs
            
        Returns:
            Dictionary containing detection results and output paths
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Create session-specific output folders
        session_output = os.path.join(self.base_output_folder, session_id)
        cropped_output_folder = os.path.join(session_output, 'cropped_predictions')
        bbox_output_folder = os.path.join(session_output, 'bbox_images')
        
        # Clean and create directories
        self.delete_directory(session_output)
        os.makedirs(cropped_output_folder, exist_ok=True)
        os.makedirs(bbox_output_folder, exist_ok=True)
        
        # Run YOLO inference
        results = self.yolo_model(image_array)
        
        if not results or len(results) == 0:
            return {
                "session_id": session_id,
                "detections": [],
                "total_detections": 0,
                "cropped_images": [],
                "bbox_images": []
            }
        
        result = results[0]
        orig_img = result.orig_img.copy()
        
        detections = []
        cropped_images = []
        bbox_images = []
        
        # Process each detection
        # Process each detection
        if result.boxes is not None:
            # Create a single image with all bounding boxes
            all_bbox_image = orig_img.copy()
            
            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                
                # Get confidence and class information if available
                confidence = float(result.boxes.conf[i]) if result.boxes.conf is not None else 0.0
                class_id = int(result.boxes.cls[i]) if result.boxes.cls is not None else 0
                class_name = result.names.get(class_id, f"class_{class_id}") if result.names else f"object_{i}"
                
                # Draw bounding box on the all_bbox_image
                cv2.rectangle(all_bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(all_bbox_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Crop the image
                cropped_image = orig_img[y1:y2, x1:x2]
                
                # Save cropped image
                cropped_filename = f"cropped_{i}_{class_name}.jpg"
                cropped_output_path = os.path.join(cropped_output_folder, cropped_filename)
                cv2.imwrite(cropped_output_path, cropped_image)
                
                # Create image with individual bounding box
                bbox_image = orig_img.copy()
                cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bbox_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Save individual bbox image
                bbox_filename = f"bbox_{i}_{class_name}.jpg"
                bbox_output_path = os.path.join(bbox_output_folder, bbox_filename)
                cv2.imwrite(bbox_output_path, bbox_image)
                
                # Store detection info
                detection_info = {
                    "id": i,
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "cropped_image_path": cropped_output_path,
                    "bbox_image_path": bbox_output_path
                }
                detections.append(detection_info)
                cropped_images.append(cropped_output_path)
                bbox_images.append(bbox_output_path)
            
            # Save the image with all bounding boxes
            all_bbox_filename = "all_detections.jpg"
            all_bbox_output_path = os.path.join(session_output, all_bbox_filename)
            cv2.imwrite(all_bbox_output_path, all_bbox_image)
        
        else:
            # No detections, save original image as all_detections
            all_bbox_output_path = os.path.join(session_output, "all_detections.jpg")
            cv2.imwrite(all_bbox_output_path, orig_img)
        
        return {
            "session_id": session_id,
            "detections": detections,
            "total_detections": len(detections),
            "cropped_images": cropped_images,
            "bbox_images": bbox_images,
            "output_folder": session_output,
            "all_detections_image": all_bbox_output_path
        }
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    
    def process_image_with_base64_output(self, image_array: np.ndarray, session_id: str = None) -> Dict[str, Any]:
        """
        Process image and return results with base64 encoded images
        """
        results = self.process_image(image_array, session_id)
        
        # Convert cropped images to base64
        for detection in results["detections"]:
            if os.path.exists(detection["cropped_image_path"]):
                detection["cropped_image_base64"] = self.image_to_base64(detection["cropped_image_path"])
            if os.path.exists(detection["bbox_image_path"]):
                detection["bbox_image_base64"] = self.image_to_base64(detection["bbox_image_path"])
        
        # Add base64 encoded version of all detections image
        if os.path.exists(results["all_detections_image"]):
            results["all_detections_image_base64"] = self.image_to_base64(results["all_detections_image"])
        
        return results
    
    def process_image_with_classification(self, image_array: np.ndarray, session_id: str = None) -> Dict[str, Any]:
        """
        Process image with YOLO detection and CNN classification
        
        Args:
            image_array: Input image as numpy array
            session_id: Optional session ID for organizing outputs
            
        Returns:
            Dictionary containing detection results with CNN classifications
        """
        # First run YOLO detection
        results = self.process_image(image_array, session_id)
        
        # Then classify each detection using CNN
        if results["detections"]:
            results["detections"] = cnn_classifier_service.classify_multiple_detections(results["detections"])
        
        return results
    
    def process_image_with_classification_and_base64(self, image_array: np.ndarray, session_id: str = None) -> Dict[str, Any]:
        """
        Process image with YOLO detection, CNN classification, and base64 encoding
        
        Args:
            image_array: Input image as numpy array
            session_id: Optional session ID for organizing outputs
            
        Returns:
            Dictionary containing detection results with CNN classifications and base64 images
        """
        # Run YOLO detection with classification
        results = self.process_image_with_classification(image_array, session_id)
        
        # Convert images to base64
        for detection in results["detections"]:
            if os.path.exists(detection["cropped_image_path"]):
                detection["cropped_image_base64"] = self.image_to_base64(detection["cropped_image_path"])
            if os.path.exists(detection["bbox_image_path"]):
                detection["bbox_image_base64"] = self.image_to_base64(detection["bbox_image_path"])
        
        # Add base64 encoded version of all detections image
        if os.path.exists(results["all_detections_image"]):
            results["all_detections_image_base64"] = self.image_to_base64(results["all_detections_image"])
        
        return results

# Global service instance
yolo_service = YOLOModelService()