import cv2
import numpy as np
from pathlib import Path
from face_detector import FaceDetector
from emoji_utils import EmojiProcessor
import time

def main():
    print("Starting Emoji Face Swap...")
    
    # Initialize paths
    BASE_DIR = Path(__file__).parent.parent
    EMOJI_DIR = BASE_DIR / "emojis"
    
    print(f"Base directory: {BASE_DIR}")
    print(f"Emoji directory: {EMOJI_DIR}")
    
    # Check if files exist
    if not EMOJI_DIR.exists():
        print(f"Error: Emoji directory not found at {EMOJI_DIR}")
        return
    
    try:
        # Initialize components
        print("Initializing FaceDetector...")
        face_detector = FaceDetector()
        
        print("Initializing EmojiProcessor...")
        emoji_processor = EmojiProcessor(str(EMOJI_DIR))
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        print("Press 'q' to quit")
        
        # For FPS calculation
        prev_frame_time = 0
        new_frame_time = 0
        
        # Track the last known good face position for smoothing
        last_face = None
        smoothing_factor = 0.5  # How much to smooth the face position (0-1)
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
                
            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Make a copy for display
            display_frame = frame.copy()
            
            try:
                # Detect faces
                faces = face_detector.detect_faces(frame)
                
                if faces:
                    # For now, just take the first face
                    rect, shape = faces[0]
                    
                    # Smooth face position if we had a previous detection
                    if last_face is not None:
                        lx, ly, lw, lh = last_face
                        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                        
                        # Apply smoothing
                        x = int(lx * (1 - smoothing_factor) + x * smoothing_factor)
                        y = int(ly * (1 - smoothing_factor) + y * smoothing_factor)
                        w = int(lw * (1 - smoothing_factor) + w * smoothing_factor)
                        h = int(lh * (1 - smoothing_factor) + h * smoothing_factor)
                        
                        # Update the rectangle
                        rect = type('obj', (object,), {
                            'left': lambda s, x=x: x,
                            'top': lambda s, y=y: y,
                            'width': lambda s, w=w: w,
                            'height': lambda s, h=h: h
                        })()
                    
                    # Store current face for next frame
                    last_face = (rect.left(), rect.top(), rect.width(), rect.height())
                    
                    try:
                        # Get face features
                        features = FaceDetector.get_face_features(shape)
                        
                        # Get the most appropriate emoji based on face features
                        emoji_name = emoji_processor.get_emoji_for_face(features)
                        
                        if emoji_name:
                            # Get face rectangle coordinates
                            x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                            
                            # Draw face rectangle (for debugging)
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            try:
                                # Overlay emoji on the face
                                display_frame = emoji_processor.overlay_emoji(
                                    display_frame, 
                                    emoji_name, 
                                    (x, y, w, h)
                                )
                                
                                # Display the predicted emotion
                                cv2.putText(display_frame, f"Emotion: {emoji_name}", 
                                          (x, y - 15), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Error overlaying emoji: {e}")
                    except Exception as e:
                        print(f"Error processing face features: {e}")
                else:
                    # No faces detected, clear the last known face
                    last_face = None
                    
                # Calculate FPS
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                
                # Display FPS
                cv2.putText(display_frame, f"FPS: {int(fps)}", 
                          (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)
                
                # Display the result
                cv2.imshow('Emoji Face Swap', display_frame)
                
                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()