import cv2
import numpy as np
import mediapipe as mp

class FaceDetector:
    def __init__(self, predictor_path=None):
        # predictor_path is kept for compatibility but not used
        print("Initializing MediaPipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("MediaPipe Face Mesh initialized")
    
    def detect_faces(self, frame):
        try:
            if frame is None:
                return []
                
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image and find faces
            results = self.face_mesh.process(rgb_frame)
            
            faces = []
            
            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                for face_landmarks in results.multi_face_landmarks:
                    # Convert landmarks to numpy array
                    landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])
                    
                    # Calculate bounding box
                    x_min = int(np.min(landmarks[:, 0]))
                    y_min = int(np.min(landmarks[:, 1]))
                    x_max = int(np.max(landmarks[:, 0]))
                    y_max = int(np.max(landmarks[:, 1]))
                    
                    # Create a rect-like object for compatibility
                    rect = type('obj', (object,), {
                        'left': lambda s, x=x_min: x,
                        'top': lambda s, y=y_min: y,
                        'width': lambda s, w=x_max-x_min: w,
                        'height': lambda s, h=y_max-y_min: h
                    })()
                    
                    faces.append((rect, landmarks))
            
            return faces
                
        except Exception as e:
            print(f"Error in detect_faces: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    @staticmethod
    def get_face_features(landmarks):
        try:
            # MediaPipe Face Mesh Indices
            # Left Eye
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            # Right Eye
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            # Lips (Outer)
            LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
            # Eyebrows
            LEFT_EYEBROW = [70, 63, 105, 66, 107]
            RIGHT_EYEBROW = [336, 296, 334, 293, 300]
            
            # Helper function for EAR
            def eye_aspect_ratio(eye_points):
                # Vertical distances
                A = np.linalg.norm(landmarks[eye_points[1]] - landmarks[eye_points[5]])
                B = np.linalg.norm(landmarks[eye_points[2]] - landmarks[eye_points[4]])
                # Horizontal distance
                C = np.linalg.norm(landmarks[eye_points[0]] - landmarks[eye_points[3]])
                return (A + B) / (2.0 * C) if C != 0 else 0

            # Helper function for MAR
            def mouth_aspect_ratio(mouth_points):
                # Vertical
                A = np.linalg.norm(landmarks[37] - landmarks[84]) # Inner lip top/bottom approx
                # Horizontal
                B = np.linalg.norm(landmarks[61] - landmarks[291]) # Corners
                return A / B if B != 0 else 0

            left_ear = eye_aspect_ratio(LEFT_EYE)
            right_ear = eye_aspect_ratio(RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Mouth features
            # Height: Top lip (0) to bottom lip (17)
            mouth_height_val = np.linalg.norm(landmarks[0] - landmarks[17])
            # Width: Left corner (61) to right corner (291)
            mouth_width_val = np.linalg.norm(landmarks[61] - landmarks[291])
            
            # Face size reference (Top of head 10 to Chin 152)
            face_height = np.linalg.norm(landmarks[10] - landmarks[152])
            face_width = np.linalg.norm(landmarks[234] - landmarks[454]) # Ear to ear approx
            face_size = (face_height + face_width) / 2.0
            
            # Normalized mouth features
            mouth_height = mouth_height_val / face_size
            mouth_width = mouth_width_val / face_size
            mar = mouth_height / mouth_width if mouth_width > 0 else 0
            mouth_ratio = mar # Using MAR as ratio
            
            # Eyebrow Raise
            # Measure distance from eye center to eyebrow
            # Left
            l_eye_center = np.mean(landmarks[LEFT_EYE], axis=0)
            l_brow_center = np.mean(landmarks[LEFT_EYEBROW], axis=0)
            l_raise = (l_eye_center[1] - l_brow_center[1]) / face_size
            
            # Right
            r_eye_center = np.mean(landmarks[RIGHT_EYE], axis=0)
            r_brow_center = np.mean(landmarks[RIGHT_EYEBROW], axis=0)
            r_raise = (r_eye_center[1] - r_brow_center[1]) / face_size
            
            eyebrow_raise = (l_raise + r_raise) / 2.0
            
            # Nose offset (not strictly needed for basic emotions but kept for compatibility)
            nose_tip = landmarks[1]
            eyes_center = (l_eye_center + r_eye_center) / 2
            nose_offset = (nose_tip[1] - eyes_center[1]) / face_size
            
            # Jaw drop (Chin 152 to Nose 1)
            jaw_drop = np.linalg.norm(landmarks[152] - landmarks[1]) / face_size
            
            # Create feature vector
            features = np.array([
                left_ear,                   # 0
                right_ear,                  # 1
                avg_ear,                    # 2
                mar,                        # 3
                mouth_height,               # 4
                mouth_ratio,                # 5
                eyebrow_raise,              # 6
                nose_offset,                # 7
                jaw_drop,                   # 8
                mouth_width                 # 9
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            print(f"Error in get_face_features: {e}")
            return np.zeros(10, dtype=np.float32)