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
        # Simple placeholder classifier; replace with real training if needed
        self.classifier = SVC(probability=True)
        X = np.random.rand(10, 10)
        y = np.random.randint(0, 2, 10)
        self.classifier.fit(X, y)

    def _load_emojis(self):
        emojis = {}
        try:
            for filename in os.listdir(self.emoji_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        name = os.path.splitext(filename)[0]
                        emoji_path = os.path.join(self.emoji_dir, filename)
                        img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            if len(img.shape) == 3 and img.shape[2] == 3:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                            emojis[name] = img
                            print(f"Successfully loaded emoji: {name}")
                        else:
                            print(f"Failed to load emoji: {filename}")
                    except Exception as e:
                        print(f"Error loading emoji {filename}: {e}")
        except Exception as e:
            print(f"Error reading emoji directory: {e}")
        return emojis

    def predict_emotion(self, face_features, debug=False):
        """Predict emotion based on facial features.
        Returns the emotion name as a string.
        """
        try:
            if len(face_features) < 9 or np.all(face_features == 0):
                if debug:
                    print("Insufficient face features, defaulting to neutral")
                return "neutral"
            left_ear, right_ear, _, mar, mouth_height, mouth_ratio, eyebrow_raise, _, _ = face_features[:9]
            eye_openness = (left_ear + right_ear) / 2.0
            mouth_openness = mar * 100
            if debug:
                print("\n--- Face Features ---")
                print(f"Eye Openness: {eye_openness:.3f} (L: {left_ear:.3f}, R: {right_ear:.3f})")
                print(f"Mouth - MAR: {mar:.3f}, Height: {mouth_height:.1f}, Ratio: {mouth_ratio:.3f}")
                print(f"Eyebrow Raise: {eyebrow_raise:.3f}, Jaw Drop: {face_features[8]:.3f}")
            emotion_scores = {'happy': 0, 'surprised': 0, 'angry': 0, 'sad': 0, 'neutral': 0}
            # Happy
            if mar > 0.35:
                emotion_scores['happy'] += 3
            elif mar > 0.25:
                emotion_scores['happy'] += 1
            if 0.1 < eye_openness < 0.28:
                emotion_scores['happy'] += 1
            # Surprised
            if eyebrow_raise > 0.045:
                emotion_scores['surprised'] += 3
            elif eyebrow_raise > 0.035:
                emotion_scores['surprised'] += 1
            if mouth_openness > 0.3:
                emotion_scores['surprised'] += 1
            if eye_openness > 0.28:
                emotion_scores['surprised'] += 1
            # Angry
            if eyebrow_raise < 0.025:
                emotion_scores['angry'] += 2
            if mar < 0.25 and mouth_ratio < 0.4:
                emotion_scores['angry'] += 1
            # Sad
            if eye_openness < 0.20:
                emotion_scores['sad'] += 1
            if mar < 0.20:
                emotion_scores['sad'] += 1
            # Neutral fallback
            if max(emotion_scores.values()) <= 1:
                emotion_scores['neutral'] = 5
            else:
                emotion_scores['neutral'] = 1
            predicted_emotion = max(emotion_scores, key=emotion_scores.get)
            if debug:
                print("\n--- Emotion Scores ---")
                for e, s in emotion_scores.items():
                    print(f"{e}: {s}")
                print(f"\nPredicted emotion: {predicted_emotion}")
            return predicted_emotion
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "neutral"

    def get_emoji_for_face(self, face_features):
        """Return the appropriate emoji image name for given face features."""
        emotion = self.predict_emotion(face_features)
        mapping = {
            "happy": "big_laugh",
            "surprised": "surprised",
            "angry": "angry",
            "sad": "sad",
            "neutral": "neutral",
        }
        # Direct match if emoji name equals emotion
        if emotion in self.emojis:
            return emotion
        # Mapped name
        mapped = mapping.get(emotion)
        if mapped and mapped in self.emojis:
            return mapped
        # Substring fallback
        for name in self.emojis:
            if emotion in name.lower():
                return name
        # Final fallback
        return next(iter(self.emojis.keys()), None)

    def overlay_emoji(self, frame, emoji_name, face_rect):
        if emoji_name not in self.emojis:
            print(f"Emoji '{emoji_name}' not found, using fallback")
            emoji_name = next(iter(self.emojis.keys()), None)
            if emoji_name is None:
                return frame
        emoji = self.emojis[emoji_name]
        x, y, w, h = face_rect
        try:
            emoji_resized = cv2.resize(emoji, (w, h))
            if emoji_resized.shape[2] == 4:
                alpha = emoji_resized[:, :, 3] / 255.0
                rgb = cv2.cvtColor(emoji_resized[:, :, :3], cv2.COLOR_RGBA2BGR)
                roi = frame[y:y+h, x:x+w]
                for c in range(3):
                    roi[:, :, c] = alpha * rgb[:, :, c] + (1 - alpha) * roi[:, :, c]
            else:
                frame[y:y+h, x:x+w] = cv2.cvtColor(emoji_resized, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error overlaying emoji: {e}")
        return frame