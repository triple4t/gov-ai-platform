import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class FaceService:
    def __init__(self):
        # Load cascades once to be reused
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def verify_faces(self, doc_image_path: str, capture_image_path: str):
        """
        Compares a reference document photo with a captured photo.
        Returns matching details and similarity score.
        """
        try:
            # 1. Read images
            img1 = cv2.imread(doc_image_path)
            img2 = cv2.imread(capture_image_path)
            
            if img1 is None or img2 is None:
                return {"error": "Images could not be loaded", "verified": False}
                
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # 2. Detect faces in both images
            faces1 = self.face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces2 = self.face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces1) == 0:
                return {"error": "No face found in reference document", "verified": False}
                    
            if len(faces2) == 0:
                return {"error": "No face found in captured image", "verified": False}
                
            # 3. Extract the first face crop from each image
            x1, y1, w1, h1 = faces1[0]
            face_crop_1 = gray1[y1:y1+h1, x1:x1+w1]
            
            x2, y2, w2, h2 = faces2[0]
            face_crop_2 = gray2[y2:y2+h2, x2:x2+w2]
            
            # To match features using MSE, images must be the exact same dimensions
            target_size = (150, 150)
            face_crop_1_resized = cv2.resize(face_crop_1, target_size)
            face_crop_2_resized = cv2.resize(face_crop_2, target_size)
            
            # Equalize histograms to improve MSE comparison across different lighting conditions
            face_crop_1_eq = cv2.equalizeHist(face_crop_1_resized)
            face_crop_2_eq = cv2.equalizeHist(face_crop_2_resized)
            
            # 4. Compare using MSE (Mean Squared Error)
            err = np.sum((face_crop_1_eq.astype("float") - face_crop_2_eq.astype("float")) ** 2)
            err /= float(face_crop_1_eq.shape[0] * face_crop_1_eq.shape[1])
            
            # Threshold adjusted for equalized histograms
            threshold = 8500 
            distance = float(err)
            
            verified = distance < threshold
            
            # Normalize score heuristically 
            similarity_score = max(0.0, min(1.0, 1.0 - (distance / 12000.0)))
            
            return {
                "verified": verified,
                "distance": distance,
                "threshold": threshold,
                "model": "opencv_mse_equalized",
                "similarity_score": similarity_score,
            }
        except Exception as e:
            logger.error(f"Error in face verification: {str(e)}")
            return {"error": str(e), "verified": False}

    def detect_liveness(self, image_path: str):
        """
        Performs a basic liveness check based on eye detection.
        Returns liveness status and score.
        """
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                 return {"is_live": False, "liveness_score": 0.0, "reason": "Could not load image"}

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) == 0:
                return {"is_live": False, "liveness_score": 0.0, "reason": "No face detected"}

            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes within the face region as a basic sign of "liveness/presence"
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            is_live = len(eyes) >= 2
            score = 0.9 if is_live else 0.4
            
            return {
                "is_live": is_live,
                "liveness_score": score,
                "eyes_detected": len(eyes),
                "reason": "Eyes visible" if is_live else "Clear eyes not visible"
            }
        except Exception as e:
            logger.error(f"Error in liveness detection: {str(e)}")
            return {"is_live": False, "liveness_score": 0.0, "error": str(e)}

face_service = FaceService()
