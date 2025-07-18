import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from typing import Dict, Any, List
import cv2

class CNNClassifierService:
    def __init__(self):
        """Initialize the CNN classifier service"""
        self.model = None
        self.img_size = (150, 150)
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model_weights', 'my_model.weights.h5')
        self._load_model()
    
    def _create_model_architecture(self):
        """Create the CNN model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
        ])
        return model
    
    def _load_model(self):
        """Load the trained CNN model"""
        try:
            # Create model architecture
            self.model = self._create_model_architecture()
            
            # Load weights if they exist
            if os.path.exists(self.model_path):
                self.model.load_weights(self.model_path)
                print(f"CNN model weights loaded from {self.model_path}")
            else:
                print(f"Warning: CNN model weights not found at {self.model_path}")
                print("Model will be available but not trained")
        except Exception as e:
            print(f"Error loading CNN model: {str(e)}")
            self.model = None
    
    def preprocess_image(self, img_array: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CNN classification
        
        Args:
            img_array: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed image array ready for prediction
        """
        # Convert BGR to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img_resized = cv2.resize(img_array, self.img_size)
        
        # Convert to float and normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def preprocess_image_from_path(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess image from file path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array ready for prediction
        """
        # Load image using Keras image utils
        img = image.load_img(image_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    
    def classify_single_detection(self, img_array: np.ndarray) -> Dict[str, Any]:
        """
        Classify a single cropped detection image
        
        Args:
            img_array: Cropped detection image as numpy array
            
        Returns:
            Dictionary containing classification results
        """
        if self.model is None:
            return {
                "error": "CNN model not loaded",
                "is_dicentric": None,
                "confidence": 0.0,
                "prediction_score": 0.0
            }
        
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(img_array)
            
            # Make prediction
            prediction = self.model.predict(processed_img, verbose=0)
            prediction_score = float(prediction[0][0])
            
            # Classify as dicentric or not
            is_dicentric = prediction_score < 0.5
            confidence = abs(prediction_score - 0.5) * 2  # Convert to confidence [0, 1]
            
            return {
                "is_dicentric": is_dicentric,
                "classification": "Dicentric" if is_dicentric else "Not Dicentric",
                "confidence": confidence,
                "prediction_score": prediction_score
            }
            
        except Exception as e:
            return {
                "error": f"Error during classification: {str(e)}",
                "is_dicentric": None,
                "confidence": 0.0,
                "prediction_score": 0.0
            }
    
    def classify_detection_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a detection from image file path
        
        Args:
            image_path: Path to the cropped detection image
            
        Returns:
            Dictionary containing classification results
        """
        if self.model is None:
            return {
                "error": "CNN model not loaded",
                "is_dicentric": None,
                "confidence": 0.0,
                "prediction_score": 0.0
            }
        
        try:
            # Load and preprocess image
            processed_img = self.preprocess_image_from_path(image_path)
            
            # Make prediction
            prediction = self.model.predict(processed_img, verbose=0)
            prediction_score = float(prediction[0][0])
            
            # Classify as dicentric or not
            is_dicentric = prediction_score < 0.5
            confidence = abs(prediction_score - 0.5) * 2  # Convert to confidence [0, 1]
            
            return {
                "is_dicentric": is_dicentric,
                "classification": "Dicentric" if is_dicentric else "Not Dicentric",
                "confidence": confidence,
                "prediction_score": prediction_score
            }
            
        except Exception as e:
            return {
                "error": f"Error during classification: {str(e)}",
                "is_dicentric": None,
                "confidence": 0.0,
                "prediction_score": 0.0
            }
    
    def classify_multiple_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple detections and return summary statistics
        
        Args:
            detections: List of detection dictionaries with cropped image paths
            
        Returns:
            List of detection dictionaries with classification results added
        """
        if self.model is None:
            print("Warning: CNN model not loaded, skipping classification")
            for detection in detections:
                detection.update({
                    "cnn_classification": {
                        "error": "CNN model not loaded",
                        "is_dicentric": None,
                        "confidence": 0.0,
                        "prediction_score": 0.0
                    }
                })
            return detections
        
        total_dicentric = 0
        total_non_dicentric = 0
        
        for detection in detections:
            # Classify using the cropped image path
            if "cropped_image_path" in detection and os.path.exists(detection["cropped_image_path"]):
                classification_result = self.classify_detection_from_path(detection["cropped_image_path"])
                
                # Add classification results to the detection
                detection["cnn_classification"] = classification_result
                
                # Count dicentric vs non-dicentric
                if classification_result.get("is_dicentric"):
                    total_dicentric += 1
                elif classification_result.get("is_dicentric") is False:
                    total_non_dicentric += 1
            else:
                detection["cnn_classification"] = {
                    "error": "Cropped image not found",
                    "is_dicentric": None,
                    "confidence": 0.0,
                    "prediction_score": 0.0
                }
        
        # Add summary to each detection for API response
        summary = {
            "total_detections": len(detections),
            "dicentric_count": total_dicentric,
            "non_dicentric_count": total_non_dicentric,
            "classification_success_rate": (total_dicentric + total_non_dicentric) / len(detections) if detections else 0
        }
        
        # Add summary to the first detection or create a separate summary
        if detections:
            detections[0]["classification_summary"] = summary
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the CNN model"""
        return {
            "model_type": "CNN Binary Classifier",
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "input_size": self.img_size,
            "classes": ["Dicentric", "Not Dicentric"],
            "architecture": "Conv2D -> MaxPool -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Dense -> Dense -> Dropout -> Dense(sigmoid)"
        }


# Global service instance
cnn_classifier_service = CNNClassifierService()
