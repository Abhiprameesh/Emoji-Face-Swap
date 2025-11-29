import cv2

def test_webcam():
    print("Testing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    print("Webcam opened successfully. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Display the frame
        cv2.imshow("Webcam Test", frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam test completed.")

if __name__ == "__main__":
    test_webcam()
