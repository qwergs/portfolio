from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")  # COCO pretrained
    model.train(
        data="chimney.yaml",
        epochs=100,
        imgsz=512,
        batch=16,
        device=0
    )

if __name__ == "__main__":
    main()
