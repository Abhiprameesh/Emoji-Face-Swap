import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class EmojiProcessor:
    def __init__(self, emoji_dir):
        self.emoji_dir = emoji_dir
        self.emojis = self._load_emojis()
        self.scaler = StandardScaler()
        self.classifier = None
        self._train_emoji_classifier()
        print(f"Loaded {len(self.emojis)} emojis")
        
    def _train_emoji_classifier(self):
        # This is a simple rule-based classifier
        # In a real application, you would train this on labeled data
        self.classifier = SVC(probability=True)
        # Dummy training data - in practice, you'd use real labeled data
        X = np.random.rand(10, 10)  # 10 samples, 10 features
        y = np.random.randint(0, 2, 10)  # 2 classes
        self.classifier.fit(X, y)
    
    def _load_emojis(self):
        emojis = {}
        try:
            for filename in os.listdir(self.emoji_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        # Use the filename without extension as the emoji name
                        name = os.path.splitext(filename)[0]
                        emoji_path = os.path.join(self.emoji_dir, filename)
                        
                        # Read image with alpha channel if present
                        img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                        
                        if img is not None:
                            # Convert to RGBA if it's RGB
                            if len(img.shape) == 3 and img.shape[2] == 3:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                            emojis[name] = img
                            print(f"Successfully loaded emoji: {name}")
                        else:
                            print(f"Failed to load emoji: {filename}")
                    except Exception as e:
                        print(f"Error loading emoji {filename}: {str(e)}")
        except Exception as e:
            print(f"Error reading emoji directory: {str(e)}")
        return emojis
    
    def predict_emotion(self, face_features):
        """
        Predict emotion based on facial features
        Returns: name of the emoji to use
        """
        try:
            # Debug print to see feature values
            print("\n--- Face Features ---")
            print(f"Left EAR: {face_features[0]:.3f}, Right EAR: {face_features[1]:.3f}")
            print(f"Mouth Aspect Ratio: {face_features[3]:.3f}, Mouth Ratio: {face_features[5]:.3f}")
            print(f"Eyebrow Raise: {face_features[6]:.3f}, Jaw Drop: {face_features[8]:.3f}")
            
            # Get features with more meaningful names
            left_ear, right_ear, avg_ear, mar, mouth_height, mouth_ratio, eyebrow_raise, nose_offset, jaw_drop = face_features[:9]
            
            # Calculate some additional features
            eye_openness = (left_ear + right_ear) / 2
            
            # Debug thresholds
            print("\n--- Threshold Checks ---")
            
            # Surprise: raised eyebrows, open mouth, wide eyes
            surprise_score = (eyebrow_raise < -5) + (jaw_drop > 30) + (eye_openness > 0.2)
            print(f"Surprise score: {surprise_score} (eyebrow_raise={eyebrow_raise:.1f}, jaw_drop={jaw_drop:.1f}, eye_openness={eye_openness:.3f})")
            
            # Happy: smile (high MAR), open mouth
            happy_score = (mar > 0.5) + (mouth_ratio > 0.1) + (eye_openness > 0.15)
            print(f"Happy score: {happy_score} (mar={mar:.3f}, mouth_ratio={mouth_ratio:.3f})")
            
            # Sad: frowning mouth, droopy eyes
            sad_score = (mar < 0.5) + (mouth_ratio > 0.08) + (eye_openness < 0.18)
            print(f"Sad score: {sad_score} (mar={mar:.3f}, eye_openness={eye_openness:.3f})")
            
            # Angry: furrowed brows, tight mouth
            angry_score = (eyebrow_raise > 5) + (mar < 0.45) + (eye_openness < 0.2)
            print(f"Angry score: {angry_score} (eyebrow_raise={eyebrow_raise:.1f}, mar={mar:.3f})")
            
            # Get the emotion with highest score
            emotions = {
                'surprised': surprise_score,
                'happy': happy_score,
                'sad': sad_score,
                'angry': angry_score,
                'neutral': 0  # Neutral is the fallback
            }
            
            # Get emotion with highest score
            predicted_emotion = max(emotions, key=emotions.get)
            
            # Only return non-neutral if we have some confidence
            if predicted_emotion != 'neutral' and emotions[predicted_emotion] < 1:
                predicted_emotion = 'neutral'
                
            print(f"\nPredicted emotion: {predicted_emotion}")
            return predicted_emotion
            
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "neutral"
    
    def get_emoji_for_face(self, face_features):
        """Get the most appropriate emoji based on face features"""
        emotion = self.predict_emotion(face_features)
        
        # Try to find an emoji that matches the emotion
        emoji_name = None
        
        # First try exact match
        if emotion in self.emojis:
            return emotion
            
        # Try variations (e.g., 'happy' might be 'smile' or 'grin')
        for name in self.emojis.keys():
            if emotion in name.lower():
                return name
                
        # If no match found, return the first available emoji or None
        return next(iter(self.emojis.keys()), None)
    
    def overlay_emoji(self, frame, emoji_name, face_rect):
        if emoji_name not in self.emojis:
            print(f"Emoji '{emoji_name}' not found in loaded emojis")
            # Try to find a fallback emoji
            emoji_name = next(iter(self.emojis.keys()), None)
            if emoji_name is None:
                return frame
        
        emoji = self.emojis[emoji_name]
        x, y, w, h = face_rect
        
        try:
            # Resize emoji to fit face
            emoji = cv2.resize(emoji, (w, h))
            
            # If emoji has alpha channel
            if emoji.shape[2] == 4:
                # Extract the alpha channel and create a mask
                alpha_s = emoji[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                
                # Convert emoji to BGR (removing alpha)
                emoji_bgr = cv2.cvtColor(emoji[:, :, :3], cv2.COLOR_RGBA2BGR)
                
                # Get the region of interest
                roi = frame[y:y+h, x:x+w]
                
                # Blend the emoji with the ROI
                for c in range(0, 3):
                    roi[:, :, c] = (alpha_s * emoji_bgr[:, :, c] + 
                                   alpha_l * roi[:, :, c])
            else:
                # If no alpha channel, just overlay
                frame[y:y+h, x:x+w] = cv2.cvtColor(emoji, cv2.COLOR_RGB2BGR)
                
        except Exception as e:
            print(f"Error overlaying emoji: {e}")
            
        return frame