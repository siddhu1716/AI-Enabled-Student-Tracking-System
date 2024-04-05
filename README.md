# Major Project 

## Overview

we have chosen the problem statement of An AI-Based Student Tracking System To Monitor His/Her Behaviour

<img src="https://github.com/siddhu1716/AI-Enabled-Student-Tracking-System/blob/main/Realtime_outputs/attendance.png" width="400" height="200"> <img src="https://github.com/siddhu1716/AI-Enabled-Student-Tracking-System/blob/main/Realtime_outputs/violence1.png" width='400' height='200'> 
<img src="https://github.com/siddhu1716/AI-Enabled-Student-Tracking-System/blob/main/Realtime_outputs/Screenshot%202024-03-21%20at%2016.45.15.png" width='400' height='200'> <img src="https://github.com/siddhu1716/AI-Enabled-Student-Tracking-System/blob/main/Realtime_outputs/vamsi_phone.png" width='400' height='200'>



Recognizing student behavior and ensuring their safety is crucial to improving the quality of teaching. Although many algorithms identify these behaviors, the results are not very accurate or efficient and mainly focus on single objects. As time has passed, new techniques have evolved recently. Object detection is gaining popularity in the field of computer vision, which also detects small objects efficiently, has better detection performance, and can effectively deal with the behavior recognition of many students. Various studies have been conducted to improve accuracy. The latest technique is YOLO v8, and when compared with other YOLO techniques, the center of an object is predicted directly instead of being offset from a known anchor box. It also introduces a more optimized network architecture and loss function

This repository contains the submission for our project by Nampalli Shiva Kumar and Team, a B. Tech CSE - AIML student from MLR Institute of Technology. The project encompasses solutions to problems demonstrating proficiency in machine learning and computer vision.

### Submitted By

- **Name:** Nampalli Shiva Kumar (Team-11)
- **Program:** B. Tech CSE - AIML
- **Institution:** MLR Institute of Technology
- **Contact:** 
  - *E-Mail:* shivanampalli@gmail.com
  - *Website:* [http://vs666.github.io](https://github.com/siddhu1716)

## Directory Structure of Submission

1. Image Attendance Contains Images of individual students to be trained for attendance purpose
2. In Models folder it contain the YOLO onnx format model.
3. The weights file consists of the best weights which are observed in training for different datasets on various parameters
4. Outputs contains the model outputs which are omitted after predicting
5. Remaining outputs like PR curve , confusion metrics , results table and all ae from YOLO
   

## Libraries Used

The project extensively utilizes the following libraries:

1.click==8.1.7

2.coloredlogs==15.0.1

3.dlib==19.24.2

4.face-recognition==1.3.0

5.face-recognition-models==0.3.0

6.flatbuffers==23.5.26

7.humanfriendly==10.0

8.mpmath==1.3.0

9.numpy==1.26.3

10.onnxruntime==1.16.3

11.opencv-python==4.9.0.80

12.packaging==23.2

13.pillow==10.2.0

14.protobuf==4.25.1

15.sympy==1.12

## How to Run Locally

To run this project on your local machine, follow these steps:

1. Clone the project repository:

   ```bash
   git clone [(https://github.com/siddhu1716/AI-Enabled-Student-Tracking-System.git)]
2. Go to the project directory

```bash
  cd /AI-Enabled-Student-Tracking-System
```

3. Install Requirements

```bash
  pip3 install -r requirements.txt
```
Keep going into task folders and run the training.ipynb notebooks

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
