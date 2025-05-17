from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Step 1: Load your trained YOLOv8 model
model = YOLO("best.pt")  # Ensure best.pt is in the same directory

# Step 2: Define expected components from your data.yaml
expected_components = ['baseplate', 'childpart1', 'childpart2', 'clinching1', 'pin1', 'pin2']

# Step 3: Load and run inference on an image
image_path = r"D:\training\MissingComponentDetector\data\test\images\top_45_part_1640592798169-293_jpg.rf.2ba71a82dfe89afae889edca84c534da.jpg"
results = model(image_path)

# Step 4: Get detected class names
detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]

# Step 5: Identify missing components
missing = list(set(expected_components) - set(detected_classes))

# Step 6: Print results
print("✅ Detected components:", detected_classes)
print("❌ Missing components:", missing)

# Step 7: Display the image with bounding boxes
annotated_image = results[0].plot()
cv2.imshow("Detected Components", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
