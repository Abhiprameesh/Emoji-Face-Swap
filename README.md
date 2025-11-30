# Emoji Face Swap

A real-time computer vision application that detects faces using your webcam and overlays expressive emojis based on facial expressions. The application uses a combination of OpenCV, dlib, and MediaPipe for face detection and facial landmark detection.

## 🎯 Features

- **Real-time Face Detection**: Utilizes dlib and OpenCV for robust face detection
- **Facial Landmark Detection**: Accurately identifies 68 facial landmarks using dlib's pre-trained model
- **Dynamic Emoji Overlay**: Smoothly overlays emojis that match your facial expressions
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Performance Optimized**: Features FPS counter and efficient processing
- **Multiple Emoji Sets**: Supports various emoji styles and expressions

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- OpenCV (cv2)
- dlib
- NumPy
- imutils
- scikit-learn
- MediaPipe
- Pillow

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
   
   Note: If you encounter issues with dlib installation on Windows, you may need to install CMake and Visual Studio Build Tools first.

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the facial landmark predictor**:
   - Download `shape_predictor_68_face_landmarks.dat.bz2` from [dlib.net/files/](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract the .dat file and place it in the `data/` directory
   - The file should be at `data/shape_predictor_68_face_landmarks.dat`

## 🖥️ Usage

Run the webcam application:
```bash
python src/run_webcam.py
```

### Controls
- `q` - Quit the application
- `c` - Capture the current frame
- `e` - Toggle between different emoji sets
- `d` - Toggle debug mode (shows facial landmarks)
- `f` - Toggle fullscreen mode

## Project Structure

```
emoji-face-swap/
├── data/                    # Data files (shape predictor model)
│   └── shape_predictor_68_face_landmarks.dat  # dlib's facial landmark predictor
├── emojis/                  # Emoji images for different expressions
│   ├── default/             # Default emoji set
│   └── custom/              # Custom emoji sets can be added here
├── models/                  # Trained models (not in version control)
├── src/                     # Source code
│   ├── emoji_utils.py       # Emoji loading, processing, and overlay logic
│   ├── face_detector.py     # Face detection and landmark detection
│   ├── run_webcam.py        # Main application script with webcam interface
│   └── train.py             # Script for training custom models (if needed)
├── .gitignore               # Git ignore file
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── setup.py                # Package configuration

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