import cv2
import numpy as np
from pathlib import Path
from imutils import face_utils
import os

class FaceDetector:
    def __init__(self, predictor_path):
        print(f"Initializing FaceDetector with predictor: {predictor_path}")
        self.use_dlib = False
        self.detector = None
        self.predictor = None
        
        # Try to import dlib
        try:
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(str(predictor_path))
            self.use_dlib = True
            print("Successfully initialized dlib face detector")
        except ImportError:
            print("Dlib not found, falling back to OpenCV face detection")
            # Initialize OpenCV's face detector with full paths
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            print(f"Loading face cascade from: {face_cascade_path}")
            print(f"Loading eye cascade from: {eye_cascade_path}")
            
            if not os.path.exists(face_cascade_path):
                print(f"Error: Face cascade file not found at {face_cascade_path}")
                # Try to find the file in other possible locations
                import glob
                possible_paths = glob.glob('**/haarcascade_*.xml', recursive=True)
                if possible_paths:
                    print(f"Found possible cascade files: {possible_paths}")
                    face_cascade_path = possible_paths[0]  # Use the first found cascade file
                    print(f"Using cascade file: {face_cascade_path}")
                else:
                    print("No cascade files found in the project directory")
            
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path) if os.path.exists(eye_cascade_path) else None
            print("Using OpenCV face detection")
        except Exception as e:
            print(f"Error initializing dlib: {e}, falling back to OpenCV")
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            
            print(f"Loading face cascade from: {face_cascade_path}")
            print(f"Loading eye cascade from: {eye_cascade_path}")
            
            if not os.path.exists(face_cascade_path):
                print(f"Error: Face cascade file not found at {face_cascade_path}")
                # Try to find the file in other possible locations
                import glob
                possible_paths = glob.glob('**/haarcascade_*.xml', recursive=True)
                if possible_paths:
                    print(f"Found possible cascade files: {possible_paths}")
                    face_cascade_path = possible_paths[0]  # Use the first found cascade file
                    print(f"Using cascade file: {face_cascade_path}")
                else:
                    print("No cascade files found in the project directory")
            
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path) if os.path.exists(eye_cascade_path) else None
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            print("Using OpenCV face detection")
    
    def detect_faces(self, frame):
        try:
            if frame is None:
                print("Error: Empty frame received")
                return []
                
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            if self.use_dlib and self.detector is not None:
                # Use dlib for detection
                rects = self.detector(gray, 0)
                faces = []
                for rect in rects:
                    try:
                        shape = self.predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                        faces.append((rect, shape))
                    except Exception as e:
                        print(f"Error processing face with dlib: {e}")
                return faces
            else:
                # Fallback to OpenCV
                faces = []
                rects = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Convert OpenCV rects to dlib-like format
                for (x, y, w, h) in rects:
                    # Create a dlib.rectangle like object
                    rect = type('obj', (object,), {
                        'left': lambda s, x=x: x,
                        'top': lambda s, y=y: y,
                        'right': lambda s, x=x, w=w: x + w,
                        'bottom': lambda s, y=y, h=h: y + h,
                        'width': lambda s, w=w: w,
                        'height': lambda s, h=h: h
                    })()
                    
                    # Create a simple face landmarks approximation
                    landmarks = np.zeros((68, 2), dtype=int)
                    # Just set some basic points as we don't have the predictor
                    landmarks[36] = [x + w//4, y + h//3]        # Left eye
                    landmarks[45] = [x + 3*w//4, y + h//3]      # Right eye
                    landmarks[30] = [x + w//2, y + h//2]        # Nose
                    landmarks[48] = [x + w//3, y + 2*h//3]      # Mouth left
                    landmarks[54] = [x + 2*w//3, y + 2*h//3]    # Mouth right
                    
                    faces.append((rect, landmarks))
                
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
        ear = (A + B) / (2.0 * C) if C != 0 else 0.0
        return ear
    
    @staticmethod
    def mouth_aspect_ratio(mouth):
        A = np.linalg.norm(mouth[13] - mouth[19])
        B = np.linalg.norm(mouth[14] - mouth[18])
        C = np.linalg.norm(mouth[15] - mouth[17])
        D = np.linalg.norm(mouth[12] - mouth[16])
        mar = (A + B + C) / (2 * D) if D != 0 else 0.0
        return mar
    
    @classmethod
    def get_face_features(cls, shape):
        try:
            # Eye aspect ratios (eye openness)
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            left_ear = cls.eye_aspect_ratio(left_eye)
            right_ear = cls.eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Mouth aspect ratio (smile/frown)
            mouth = shape[48:68]
            mar = cls.mouth_aspect_ratio(mouth)
            
            # Mouth height and width ratios
            mouth_height = np.linalg.norm(shape[62] - shape[66])
            mouth_width = np.linalg.norm(shape[54] - shape[48])
            mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0.1
            
            # Eyebrow raise (for surprise/anger)
            left_eyebrow = np.mean(shape[17:22], axis=0)
            right_eyebrow = np.mean(shape[22:27], axis=0)
            
            # Calculate eyebrow raise relative to eye position
            left_eye_center = np.mean(shape[36:42], axis=0)
            right_eye_center = np.mean(shape[42:48], axis=0)
            
            # Eyebrow raise (negative values mean raised eyebrows)
            left_eyebrow_raise = left_eye_center[1] - left_eyebrow[1]
            right_eyebrow_raise = right_eye_center[1] - right_eyebrow[1]
            eyebrow_raise = (left_eyebrow_raise + right_eyebrow_raise) / 2.0
            
            # Nose position (relative to eyes)
            nose_tip = shape[33]  # Using nose tip instead of point 30 for better stability
            eyes_center = (left_eye_center + right_eye_center) / 2
            nose_offset = nose_tip[1] - eyes_center[1]
            
            # Jaw drop (for surprise) - using chin to nose tip distance
            jaw_drop = shape[8][1] - nose_tip[1]
            
            # Normalize features based on face size
            face_width = np.linalg.norm(shape[16] - shape[0])  # Distance between face edges
            if face_width > 0:
                mouth_ratio /= face_width
                jaw_drop /= face_width
                eyebrow_raise /= face_width
            
            # Create feature vector with meaningful values
            features = [
                left_ear,                   # Left eye aspect ratio
                right_ear,                  # Right eye aspect ratio
                avg_ear,                    # Average eye aspect ratio
                mar,                        # Mouth aspect ratio
                mouth_height,               # Absolute mouth height
                mouth_ratio,                # Mouth height/width ratio
                eyebrow_raise,              # Eyebrow raise (negative = raised)
                nose_offset,                # Nose position relative to eyes
                jaw_drop,                   # Jaw drop (chin to nose tip)
                mouth_width                 # Mouth width
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in get_face_features: {e}")
            # Return default values in case of error
            return np.zeros(10, dtype=np.float32)