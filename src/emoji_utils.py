import cv2
import numpy as np
import os

class EmojiProcessor:
    def __init__(self, emoji_dir):
        print(f"Initializing EmojiProcessor with directory: {emoji_dir}")
        self.emojis = self.load_emojis(emoji_dir)
        print(f"Loaded {len(self.emojis)} emojis")
    
    @staticmethod
    def load_emojis(emoji_dir):
        emojis = {}
        if not os.path.exists(emoji_dir):
            print(f"Error: Emoji directory {emoji_dir} does not exist")
            return emojis
            
        for filename in os.listdir(emoji_dir):
            if filename.endswith('.png'):
                name = os.path.splitext(filename)[0]
                path = os.path.join(emoji_dir, filename)
                try:
                    emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if emoji is not None:
                        emojis[name] = emoji
                        print(f"Loaded emoji: {name}")
                    else:
                        print(f"Failed to load emoji: {filename}")
                except Exception as e:
                    print(f"Error loading emoji {filename}: {e}")
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
            
            # Overlay emoji on face
            if emoji.shape[2] == 4:  # If emoji has alpha channel
                alpha = emoji[:, :, 3] / 255.0
                overlay_color = emoji[:, :, :3]
                
                for c in range(3):
                    frame[y:y+h, x:x+w, c] = (
                        (1 - alpha) * frame[y:y+h, x:x+w, c] +
                        alpha * overlay_color[:, :, c]
                    )
            else:
                frame[y:y+h, x:x+w] = emoji
                
        except Exception as e:
            print(f"Error overlaying emoji: {e}")
            
        return frame