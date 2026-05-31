from ultralytics import YOLO

# Load DotNeuralNet's pretrained Braille weights

model = YOLO("model/yolov8_braille.pt")

# Train on braillify dataset
model.train(
    data="dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    name="braille_model",
    project="training/results"
)

print("Training complete!")
print("Best weights saved at: training/results/braille_model/weights/best.pt")