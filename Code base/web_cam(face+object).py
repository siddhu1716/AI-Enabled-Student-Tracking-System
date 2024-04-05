import cv2
import numpy as np
import face_recognition
from yolov8 import YOLOv8
import os
from datetime import datetime

# Initialize YOLOv8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

def load_known_images(path):
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    return images, classNames

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encode = face_encodings[0]
            encodeList.append(encode)
        else:
            encodeList.append(None)
    return encodeList

def mark_attendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def perform_face_recognition(frame, encodeListKnown, classNames):
    resized_frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)  # Resize frame for faster processing
    
    imgS = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2  # Scale up bounding box coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    return frame

# Load known images and their encodings
known_images, class_names = load_known_images('ImagesAttendance')
encode_list_known = find_encodings(known_images)
print('Encoding Complete')

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab a frame.")
        break

    boxes, _, _ = yolov8_detector(frame)
    combined_img = yolov8_detector.draw_detections(frame)

    result_img = perform_face_recognition(combined_img, encode_list_known, class_names)

    cv2.imshow("Combined Detection", result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
