from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction
results = model.predict(source="E:/dl project/WhatsApp Image 2025-09-02 at 09.17.42_6add6e63.jpg", conf=0.3, save=True)

# Loop through predictions and display with annotations
for r in results:
    annotated_img = r.plot()  
    cv2.imshow("YOLO Predictions", annotated_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
