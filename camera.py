import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("camera failed to open")
    exit()

print("model is working, press q to exit")

while True:

    ret, frame = cap.read()
    if not ret:
        print("failed to capture image")
        break
    frame = cv2.flip(frame, 1)

    results = model(frame, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("face detection", annotated_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("program closed")
