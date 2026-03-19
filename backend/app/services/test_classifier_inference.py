from pathlib import Path
from app.services.image_classifier_cnn import classify_ecg_image

def main():
    sample = next(Path("data/image_training/classifier/test/normal").glob("*.png"))
    result = classify_ecg_image(sample.read_bytes())
    print("Imagen:", sample)
    print(result)

if __name__ == "__main__":
    main()