import cv2
import pickle
import numpy as np
from pathlib import Path
from face_detector import FaceDetector
from emoji_utils import EmojiProcessor

def main():
    print("Starting Emoji Face Swap...")
    
    # Initialize paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_PATH = BASE_DIR / "models" / "emoji_classifier.pkl"
    PREDICTOR_PATH = BASE_DIR / "data" / "shape_predictor_68_face_landmarks.dat"
    EMOJI_DIR = BASE_DIR / "emojis"
    
    print(f"Base directory: {BASE_DIR}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Predictor path: {PREDICTOR_PATH}")
    print(f"Emoji directory: {EMOJI_DIR}")
    
    # Check if files exist
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
        
    if not PREDICTOR_PATH.exists():
        print(f"Error: Predictor file not found at {PREDICTOR_PATH}")
        return
        
    if not EMOJI_DIR.exists():
        print(f"Error: Emoji directory not found at {EMOJI_DIR}")
        return
    
    try:
        # Initialize components
        print("Initializing FaceDetector...")
        face_detector = FaceDetector(str(PREDICTOR_PATH))
        
        print("Initializing EmojiProcessor...")
        emoji_processor = EmojiProcessor(str(EMOJI_DIR))
        
        # Load the trained model
        print("Loading model...")
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        clf = model_data['classifier']
        scaler = model_data['scaler']
        emoji_labels = model_data['labels']
        print(f"Model loaded with {len(emoji_labels)} emoji classes")
        
        # Initialize webcam
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        print("Webcam initialized. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
                
            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Make sure the frame is in the correct format (BGR)
            if frame is None:
                print("Error: Empty frame from webcam")
                continue
                
            # Convert to BGR if it's not already (some webcams might return RGB)
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif frame.shape[2] == 3:  # Already BGR or RGB
                # Ensure it's BGR (some webcams might return RGB)
                if frame[0,0,0] > frame[0,0,2]:  # If B > R, might be BGR
                    pass  # Already BGR
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Make a copy of the frame for display
            display_frame = frame.copy()
            
            # Detect faces
            faces = face_detector.detect_faces(frame)
            
            for rect, shape in faces:
                try:
                    # Get face rectangle
                    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                    
                    # Extract features
                    features = FaceDetector.get_face_features(shape)
                    features_scaled = scaler.transform([features])
                    
                    # Predict emoji
                    pred = clf.predict(features_scaled)[0]
                    emoji_name = emoji_labels[pred]
                    print(f"Predicted emoji: {emoji_name}")
                    
                    # Overlay emoji on the display frame
                    display_frame = emoji_processor.overlay_emoji(display_frame, emoji_name, (x, y, w, h))
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Display the result
            cv2.imshow("Emoji Face Swap", display_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()