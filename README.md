# Emoji Face Swap

A real-time computer vision application that detects faces using your webcam and overlays expressive emojis based on facial expressions. The application uses a combination of OpenCV and dlib (with a fallback to OpenCV's Haar cascades) for face detection and emotion recognition.

##  Features

- **Real-time Face Detection**: Uses dlib (with OpenCV fallback) for robust face detection, but as of now only OpenCV is working properly
- **Facial Expression Analysis**: Detects emotions like happy, sad, angry, and surprised
- **Dynamic Emoji Overlay**: Smoothly overlays emojis that match your facial expressions
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Responsive UI**: Shows FPS counter and current emotion detection
- **Multiple Emoji Sets**: Supports various emoji styles and expressions

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenCV (cv2)
- dlib (optional, falls back to OpenCV if not available)
- NumPy
- imutils
- scikit-learn

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/emoji-face-swap.git
   cd emoji-face-swap
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the facial landmark predictor**:
   - Download `shape_predictor_68_face_landmarks.dat` from [dlib.net/files/](http://dlib.net/files/)
   - Place it in the `data/` directory

## 🖥️ Usage

Run the webcam application:
```bash
python src/run_webcam.py
```

### Controls
- `q` - Quit the application
- `c` - Capture the current frame
- `e` - Toggle between different emoji sets

## Project Structure

```
emoji-face-swap/
├── data/                    # Data files (shape predictor model)
├── emojis/                  # Emoji images for different expressions
├── models/                  # Trained models (not in version control)
├── src/                     # Source code
│   ├── emoji_utils.py       # Emoji loading and overlay logic
│   ├── face_detector.py     # Face detection and feature extraction
│   ├── run_webcam.py        # Main application script
│   └── train.py             # Model training script
├── .gitignore               # Git ignore file
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── setup.py                # Package configuration
```

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- [dlib](http://dlib.net/) - For facial landmark detection
- [OpenCV](https://opencv.org/) - For computer vision utilities
- [Twemoji](https://twemoji.twitter.com/) - For emoji designs (example source)
- [imutils](https://github.com/jrosebr1/imutils) - For computer vision convenience functions