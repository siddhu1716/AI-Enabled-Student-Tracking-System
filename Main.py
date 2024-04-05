import cv2
import numpy as np
import face_recognition
from keras.models import load_model
from yolov8 import YOLOv8
import os
from datetime import datetime
import imutils as im
from twilio.rest import Client

account_sid='AC8836dfeb51a6f5ea7f0b97cf4e7b2696'

auth_token = 'ade6965347c16992392b4af33c216e2d'

twilio_phone_number = 'whatsapp:+14155238886'

recipient_phone_numbers = ['whatsapp:+919502152068', 'whatsapp:+919000608068', 'whatsapp:+919441841865']

# Create a Twilio client
client = Client(account_sid, auth_token)

def send_whatsapp_message(message_content,to):
    # Send the message
    message = client.messages.create(
        body=message_content,
        from_=twilio_phone_number,
        to=to
    )

    # Print the message SID (optional)
    print(f"Message SID: {message.sid}")

event_message_content = "There is an abnormal event occured"
message_sent = False


# Initialize YOLOv8 object detector
model_path = "/Users/apple/Desktop/Student_tracking_system/models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

model=load_model('/Users/apple/Desktop/Student_tracking_system/saved_model 20-01-22.h5')

def load_known_images(path):
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        if cl.endswith('.DS_Store'):
            continue  # Skip .DS_Store files
        curImg = cv2.imread(f'{path}/{cl}')
        if curImg is None:
            print(f"Error: Unable to read image '{cl}'")
            continue
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    return images, classNames

def find_encodings(images):
    encodeList = []
    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # it, converts BGR format to RGB format,
            face_encodings = face_recognition.face_encodings(img)
        # uses the face_encodings function from the face_recognitionlibrary to extract facial encodings for each image. 

        # The facial encodings are stored in a new list calledencodeList`, which is returned by the function.
            if len(face_encodings) > 0:
                encode = face_encodings[0]
                encodeList.append(encode)
            else:
                encodeList.append(None)
            # This line appends None to the encodeList list to indicate that there are no facial encodings in the current image.
        except:
            encodeList.append(None)
    return encodeList


def mark_attendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def mobile_phone_near_face(face_y1, face_x1, face_y2, face_x2, yolov8_detector):
    mobile_class_id = 67  # Adjust if needed for your YOLOv8 model
    mobile_threshold = 0.2  # Adjust as needed
    boxes, _, class_ids = yolov8_detector.detect_objects(frame)

    for box, class_id in zip(boxes, class_ids):
        if class_id == mobile_class_id:
            box_y1, box_x1, box_y2, box_x2 = box
            
            # Check for overlap between face and mobile bounding boxes
            if (
                box_x1 < face_x2 + mobile_threshold * (face_x2 - face_x1) and
                box_x2 > face_x1 - mobile_threshold * (face_x2 - face_x1) and
                box_y1 < face_y2 + mobile_threshold * (face_y2 - face_y1) and
                box_y2 > face_y1 - mobile_threshold * (face_y2 - face_y1)
            ):
                return True  # Mobile detected near the face

    return False 


def compare_face_encodings(known_encodings, face_to_compare, tolerance=0.6):
    # Remove None values from known_encodings
    known_encodings = [encoding for encoding in known_encodings if encoding is not None]

    if not known_encodings or face_to_compare is None:
        return False  # If no valid face encodings or face_to_compare is None, consider it a mismatch

    # Compare the face encoding to each known encoding
    matches = [face_recognition.compare_faces([known_encoding], face_to_compare, tolerance=tolerance)[0] for known_encoding in known_encodings]
    
    return any(matches) 

def perform_face_recognition(frame, encodeListKnown, classNames):
    resized_frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)  # Resize frame for faster processing
    
    imgS = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        name = "Unknown"
    
        if compare_face_encodings(encodeListKnown, encodeFace):

            encodeFace = [encodeFace]
            encodeListKnown = [enc for enc in encodeListKnown if enc is not None and len(enc) == 128]           
            encodeListKnown = np.array(encodeListKnown)
            print(len(encodeListKnown))
            print(len(encodeFace))
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


def mean_squared_loss(x1,x2):
    difference=x1-x2
    a,b,c,d,e=difference.shape
    n_samples=a*b*c*d*e
    sq_difference=difference**2
    Sum=sq_difference.sum()
    distance=np.sqrt(Sum)
    mean_distance=distance/n_samples
    return mean_distance

# Load known images and their encodings
known_images, class_names = load_known_images('ImagesAttendance')
encode_list_known = find_encodings(known_images)
print('Encoding Complete')


cap = cv2.VideoCapture(0)
# cap=cv2.VideoCapture('/Users/apple/Desktop/Student_tracking_system/24.mp4')

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
while cap.isOpened():
    imagedump=[]
    ret,frame=cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break
    
    # Add error handling for empty frames
    if frame is None:
        print("Error: Empty frame received.")
        continue
    
    boxes, _, _ = yolov8_detector(frame)
    combined_img = yolov8_detector.draw_detections(frame)

    result_img = perform_face_recognition(combined_img, encode_list_known, class_names)

    for i in range(10):
        ret,frame=cap.read()
        if ret == False:
            break
        image = im.resize(frame, width=1000, height=1000, inter=cv2.INTER_AREA)
        frame = cv2.resize(frame, (227, 227), interpolation=cv2.INTER_AREA)
        gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
        gray = (gray - gray.mean()) / gray.std()
        gray = np.clip(gray, 0, 1)
        imagedump.append(gray)
    imagedump=np.array(imagedump)
    imagedump.resize(227,227,10)
    imagedump=np.expand_dims(imagedump,axis=0)
    imagedump=np.expand_dims(imagedump,axis=4)
    output=model.predict(imagedump)
    loss=mean_squared_loss(imagedump,output)
    if ret == False:
        print("video end")
        break
    if cv2.waitKey(15) & 0xFF==ord('q'):
        break
    print(loss)

    if loss > 0.00064 and not message_sent:
        print('Abnormal Event Detected')
        for recipient in recipient_phone_numbers:
            send_whatsapp_message(event_message_content, recipient)
        message_sent = True

        cv2.putText(result_img, "Abnormal Event", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (250, 24, 255), 4)

    cv2.imshow("video",result_img)

cap.release()
cv2.destroyAllWindows()
