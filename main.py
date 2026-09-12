import cv2
import numpy as np
import torch


def load_model():
    """Load Pre-trained YOLOv5 model."""
    print("Loading YOLOv5 model...")
    return torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def detect_objects(image_path, model):
    """Detect objects in a static image file."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at path '{image_path}'.")
        return

    results = model(image)
    results.show()


def detect_from_webcam(model):
    """Detect objects in real time using the system camera."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    print("Starting webcam stream... Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            break

        # Inference
        results = model(frame)

        # Convert YOLO output (RGB) back into OpenCV format (BGR)
        rendered_img_rgb = results.render()[0]
        rendered_img_bgr = cv2.cvtColor(rendered_img_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow("Real-Time Object Detection", rendered_img_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo_model = load_model()

    print(
        "\nChoose an option: \n1. Detect object in an image\n2. Real-time object detection from webcam"
    )
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        path = input("Enter image file path: ").strip()
        detect_objects(path, yolo_model)
    elif choice == "2":
        detect_from_webcam(yolo_model)
    else:
        print("Invalid Option.")
    

  
 

    
  
    
  
