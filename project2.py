from ultralytics import YOLO

def train_weapon_model():
    # Load YOLOv8 pretrained model
    model = YOLO("yolov8n.pt")   # you can use yolov8s.pt / yolov8m.pt for bigger models

    # Train the model
    model.train(
        data="E:/dl project/weapon_dataset/data.yaml",  # path to your dataset yaml
        epochs=30,
        imgsz=640,
        workers=0   # 👈 avoids multiprocessing issues on Windows
    )

    # Validate model performance
    metrics = model.val()
    print("Validation metrics:", metrics)

    # Run inference on test images
    results = model.predict(
        source="E:/dl project/weapon_dataset/images/test",  # folder of test images
        save=True,    # saves annotated images in runs/detect/predict
        conf=0.25     # confidence threshold
    )

    # Show predictions one by one
    for r in results:
        r.show()   # opens window with annotations (if supported)

if __name__ == "__main__":   # 👈 IMPORTANT for Windows
    train_weapon_model()
