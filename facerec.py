import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Load known images and their encodings
def load_known_images(path):
    images = []
    classNames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                curImg = cv2.imread(os.path.join(root, file))
                images.append(curImg)
                classNames.append(os.path.splitext(file)[0])
    return images, classNames

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if len(encode) > 0:
            encodeList.append(encode[0])
        else:
            encodeList.append(None)
    return encodeList

known_images, class_names = load_known_images('ImagesAttendance')
encode_list_known = find_encodings(known_images)
print('Encoding Complete')

# Function to mark attendance
def mark_attendance(name):
    with open('Attendance.csv', 'a') as f:
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name},{dtString}\n')

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # Find faces in the frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        # Compare each face encoding with known encodings
        matches = face_recognition.compare_faces(encode_list_known, face_encoding, tolerance=0.6)
        name = "Unknown"

        # Check for match
        if True in matches:
            match_index = matches.index(True)
            name = class_names[match_index]
            mark_attendance(name)

        # Draw rectangle and label
        top, right, bottom, left = face_locations[matches.index(True)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
