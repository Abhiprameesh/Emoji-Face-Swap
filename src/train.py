import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from face_detector import FaceDetector
from emoji_utils import EmojiProcessor

def train_model(emoji_dir, predictor_path, output_path):
    # Initialize components
    face_detector = FaceDetector(predictor_path)
    emoji_processor = EmojiProcessor(emoji_dir)
    
    # Collect training data
    X = []
    y = []
    emoji_labels = {}
    
    print("Collecting training data...")
    
    # For each emoji, create training samples
    for i, (emoji_name, emoji) in enumerate(emoji_processor.emojis.items()):
        emoji_labels[i] = emoji_name
        print(f"Processing {emoji_name}...")
        
        # Save emoji to temp file for detection
        temp_path = "temp_emoji.png"
        cv2.imwrite(temp_path, emoji)
        
        # Load and process emoji
        emoji_img = cv2.imread(temp_path)
        if emoji_img is None:
            print(f"Failed to load {emoji_name}")
            continue
            
        # Detect faces in emoji
        faces = face_detector.detect_faces(emoji_img)
        
        if not faces:
            print(f"No face detected in {emoji_name}")
            continue
            
        # Use the first detected face
        _, shape = faces[0]
        
        # Extract features
        features = face_detector.get_face_features(shape)
        X.append(features)
        y.append(i)
        
        # Add some synthetic variations
        for _ in range(5):
            # Random transformations
            angle = np.random.uniform(-15, 15)
            scale = np.random.uniform(0.9, 1.1)
            
            M = cv2.getRotationMatrix2D((emoji.shape[1]/2, emoji.shape[0]/2), angle, scale)
            transformed = cv2.warpAffine(emoji, M, (emoji.shape[1], emoji.shape[0]), 
                                        borderMode=cv2.BORDER_CONSTANT, 
                                        borderValue=(0, 0, 0, 0))
            
            cv2.imwrite(temp_path, transformed)
            emoji_img = cv2.imread(temp_path)
            
            if emoji_img is not None:
                faces = face_detector.detect_faces(emoji_img)
                if faces:
                    _, shape = faces[0]
                    features = face_detector.get_face_features(shape)
                    X.append(features)
                    y.append(i)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    if not X:
        print("No valid training data found!")
        return
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train classifier
    print("Training classifier...")
    clf = KNeighborsClassifier(n_neighbors=min(3, len(emoji_labels)))
    clf.fit(X, y)
    
    # Save model
    model_data = {
        'classifier': clf,
        'scaler': scaler,
        'labels': emoji_labels
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train emoji classifier')
    parser.add_argument('--emoji_dir', type=str, default='../emojis',
                        help='Directory containing emoji images')
    parser.add_argument('--predictor_path', type=str, 
                        default='../data/shape_predictor_68_face_landmarks.dat',
                        help='Path to dlib shape predictor')
    parser.add_argument('--output_path', type=str,
                        default='../models/emoji_classifier.pkl',
                        help='Path to save the trained model')
    
    args = parser.parse_args()
    train_model(args.emoji_dir, args.predictor_path, args.output_path)