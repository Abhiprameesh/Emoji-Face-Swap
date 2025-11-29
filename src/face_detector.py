import cv2
import dlib
import numpy as np
from pathlib import Path
from imutils import face_utils

class FaceDetector:
    def __init__(self, predictor_path):
        print(f"Initializing FaceDetector with predictor: {predictor_path}")
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(str(predictor_path))
            print("FaceDetector initialized successfully")
        except Exception as e:
            print(f"Error initializing FaceDetector: {e}")
            raise
    
    def detect_faces(self, frame):
        try:
            # Check if frame is valid
            if frame is None:
                print("Error: Empty frame received")
                return []
                
            print(f"Input frame shape: {frame.shape}, dtype: {frame.dtype}")
            
            # Make a copy to avoid modifying the original
            img = frame.copy()
            
            # Convert to 8-bit if needed
            if img.dtype != np.uint8:
                print(f"Converting image from {img.dtype} to uint8")
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Handle different color spaces
            if len(img.shape) == 2:  # Already grayscale
                gray = img
            elif img.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:  # BGR or RGB
                # First convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                print(f"Unsupported image format. Shape: {img.shape}, dtype: {img.dtype}")
                return []
                
            print(f"Grayscale image shape: {gray.shape}, dtype: {gray.dtype}, min: {gray.min()}, max: {gray.max()}")
            
            # Ensure grayscale is 2D
            if len(gray.shape) > 2:
                gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            
            # Normalize to 0-255 if needed
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            
            # Detect faces
            print(f"Detecting faces in image of shape {gray.shape}, dtype: {gray.dtype}")
            rects = self.detector(gray, 0)  # 0 means no upscaling
            faces = []
            
            for rect in rects:
                try:
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    faces.append((rect, shape))
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
                
            if not faces:
                print("No faces detected in the image")
            return faces
            
        except Exception as e:
            print(f"Error in detect_faces: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    
    @staticmethod
    def mouth_aspect_ratio(mouth):
        A = np.linalg.norm(mouth[13] - mouth[19])
        B = np.linalg.norm(mouth[14] - mouth[18])
        C = np.linalg.norm(mouth[15] - mouth[17])
        D = np.linalg.norm(mouth[12] - mouth[16])
        mar = (A + B + C) / (2 * D)
        return mar
    
    @classmethod
    def get_face_features(cls, shape):
        features = []
        
        # Eye aspect ratios
        left_eye = shape[42:48]
        right_eye = shape[36:42]
        left_ear = cls.eye_aspect_ratio(left_eye)
        right_ear = cls.eye_aspect_ratio(right_eye)
        features.extend([left_ear, right_ear])
        
        # Mouth aspect ratio
        mouth = shape[48:68]
        mar = cls.mouth_aspect_ratio(mouth)
        features.append(mar)
        
        # Mouth height
        mouth_height = np.linalg.norm(shape[62] - shape[66])
        features.append(mouth_height)
        
        # Eyebrow raise
        left_eyebrow = np.mean(shape[17:22], axis=0)
        right_eyebrow = np.mean(shape[22:27], axis=0)
        eyebrow_raise = (left_eyebrow[1] + right_eyebrow[1]) / 2
        features.append(eyebrow_raise)
        
        return np.array(features)