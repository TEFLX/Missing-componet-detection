from ultralytics import YOLO
import os

# ✅ Use a relative or raw path to avoid Windows escape character issues
DATA_YAML_PATH = r'D:/training/MissingComponentDetector/data/data.yaml'  # Make sure this is the correct relative path from your project root

def fix_yaml_paths():
    """Fix Roboflow default paths inside data.yaml if needed."""
    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ {DATA_YAML_PATH} not found.")
        return

    with open(DATA_YAML_PATH, "r") as f:
        content = f.read()

    # Replace Roboflow-style relative paths with local paths
    content = content.replace("../train/images", "data/train/images")
    content = content.replace("../valid/images", "data/valid/images")
    content = content.replace("../test/images", "data/test/images")

    with open(DATA_YAML_PATH, "w") as f:
        f.write(content)

    print("✅ Fixed paths in data.yaml")

def train_yolov8():
    """Train the YOLOv8 model on local machine (CPU by default)"""
    model = YOLO("yolov8n.yaml")  # You can use yolov8s.yaml if you want better accuracy and have more RAM

    model.train(
        data=DATA_YAML_PATH,
        epochs=30,
        imgsz=640,
        batch=4,         # Set small batch size for CPU systems
        device='cpu',    # Change to 'cuda' if you are using Colab or have a GPU
        name="missing_component_model"
    )

    # Save best weights to models folder
    model.save("models/best.pt")
    print("✅ Training complete. Model saved to models/best.pt")

if __name__ == '__main__':
    fix_yaml_paths()
    train_yolov8()
