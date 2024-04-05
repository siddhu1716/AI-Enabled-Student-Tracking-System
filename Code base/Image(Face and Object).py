import cv2
import numpy as np
import face_recognition
from yolov8 import YOLOv8
import os
from datetime import datetime

# Initialize YOLOv8 object detector
model_path = "models/yolov8m.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

def perform_face_recognition_and_detection(image):
    path = 'ImagesAttendance'
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                encodeList.append(face_encodings[0])
        return encodeList

    encodeListKnown = findEncodings(images)

    # Perform object detection using YOLOv8
    boxes, _, class_ids = yolov8_detector(image)
    combined_img = yolov8_detector.draw_detections(image)

    # Perform face recognition on the input image
    imgS = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(combined_img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(combined_img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return combined_img  # Return the image with recognized faces marked and objects detected

# Load an image for processing
input_image_path = '/Users/apple/Desktop/images/IMG_6074.jpg'

# Read the input image
input_image = cv2.imread(input_image_path)

# Check if the image is read properly
if input_image is not None:
    result_image = perform_face_recognition_and_detection(input_image)
    cv2.imshow("Combined Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not read the input image.")
