# Emoji Face Swap

A fun computer vision application that detects faces in real-time using a webcam and replaces them with emojis based on facial expressions.

## Features

- Real-time face detection using OpenCV and dlib
- Facial landmark detection for accurate emoji placement
- Dynamic emoji selection based on facial expressions
- Smooth emoji overlay with transparency
- Webcam integration for live demonstration

## Prerequisites

- Python 3.7+
- OpenCV
- dlib
- NumPy
- imutils
- scikit-learn (for emotion classification)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emoji-face-swap.git
   cd emoji-face-swap
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On Unix or MacOS
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the facial landmark predictor (shape_predictor_68_face_landmarks.dat) and place it in the `data/` directory.

## Usage

Run the webcam demo:
```bash
python -m src.run_webcam
```

### Controls
- Press 'q' to quit the application
- Press 'c' to capture the current frame
- Press 'e' to toggle between different emoji sets

## Project Structure

```
emoji-face-swap/
├── data/                    # Data files (e.g., shape predictor model)
├── emojis/                  # Emoji images
├── models/                  # Trained models
├── src/                     # Source code
│   ├── emoji_utils.py       # Emoji handling utilities
│   ├── face_detector.py     # Face detection and feature extraction
│   └── run_webcam.py        # Main application script
├── .gitignore
├── README.md
└── requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- dlib for facial landmark detection
- OpenCV for computer vision utilities
- Emoji designs from [Twemoji](https://twemoji.twitter.com/)