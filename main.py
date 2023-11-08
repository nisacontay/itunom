from ultralytics import YOLO
import cv2
import math 
# webcam oluşturma
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# obje classları
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

#nesne tespitinin görüntülenmesi
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    #kalıcı kutu
    height, width, _ = img.shape
    print(height, width)
    h2= int(height * 0.10)
    h1 = int(height * 0.90)
    w1 = int(width * 0.25)
    w2 = int(width * 0.75)
    print(h1, h2, w1, w2)

    cv2.rectangle(img, (w1, h1), (w2, h2), (128, 0, 0), 2)
    
    # nesneleri işaretleme
    for r in results:
        boxes = r.boxes
       
        for box in boxes:

            # koordinatlar
            x1, x2 , y1 , y2 = box.xyxy[0]
            x1, x2 , y1, y2 = int(x1), int(x2), int(y1), int(y2)
            print(x1, x2, y1, y2)

            #dikdörtgenleme
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # nesne bilgileri
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # programı durdurma
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()