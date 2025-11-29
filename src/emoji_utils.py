import os
import cv2
import numpy as np
from pathlib import Path

class EmojiProcessor:
    def __init__(self, emoji_dir):
        self.emoji_dir = emoji_dir
        self.emojis = self._load_emojis()
        print(f"Loaded {len(self.emojis)} emojis")
    
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
    
    def overlay_emoji(self, frame, emoji_name, face_rect):
        if emoji_name not in self.emojis:
            print(f"Emoji '{emoji_name}' not found in loaded emojis")
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