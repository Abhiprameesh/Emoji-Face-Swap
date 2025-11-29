import os
import sys
import subprocess
import urllib.request
import bz2

def download_file(url, filename):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filename)
    return filename

def extract_bz2(filename):
    print(f"Extracting {filename}...")
    with bz2.BZ2File(filename, 'rb') as f_in:
        with open(filename[:-4], 'wb') as f_out:
            f_out.write(f_in.read())
    os.remove(filename)
    return filename[:-4]

def setup_environment():
    print("Setting up Emoji Face Swap...")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("emojis", exist_ok=True)
    
    # Download shape predictor if not exists
    predictor_path = "data/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_file = download_file(url, "shape_predictor_68_face_landmarks.dat.bz2")
        extract_bz2(bz2_file)
        os.rename("shape_predictor_68_face_landmarks.dat", predictor_path)
    
    print("\nSetup complete!")
    print("Please add your emoji images to the 'emojis' folder.")
    print("Then run 'python src/run_webcam.py' to start the application.")

if __name__ == "__main__":
    print("=" * 50)
    print("Emoji Face Swap - Setup")
    print("=" * 50)
    setup_environment()