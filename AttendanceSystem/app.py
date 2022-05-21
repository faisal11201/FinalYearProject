# Required Libraries
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# Flask Integration
app = Flask(__name__)

# camera for live-streaming
camera = cv2.VideoCapture(0)


# Load a sample picture and learn how to recognize it.
def findEncodings(inputImages):
    encodeList = []
    for img in inputImages:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    roll, name = name.split('$')
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        rollList = []
        for line in myDataList:
            entry = line.split(',')
            attr = entry[0].split('$')
            rollList.append(attr[0])

        if roll not in rollList:
            now = datetime.now()
            date = now.strftime('%H:%M:%S')
            f.writelines(f'\n{roll},{name},{date}')


# path to folder of input images
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Create arrays of known face encodings and their names
known_face_encodings = encodeListKnown
known_face_names = classNames


def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time

            # Find all the faces and face encodings in the current frame of video
            faceLocations = face_recognition.face_locations(rgb_small_frame)
            faceEncodings = face_recognition.face_encodings(rgb_small_frame, faceLocations)
            faceNames = []
            for face_encoding in faceEncodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    markAttendance(name)

                faceNames.append(name)

            # Display the results
            for (top, right, bottom, left), name in zip(faceLocations, faceNames):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def result():
    return render_template('result.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

# -----------------  How to run ------------------------------------------------------------------------------------
# Step 1:  Enter command `python main.py` or hit run widget(if using IDE)
# Step 2: Open : http://127.0.0.1:5000 in browser
# Step 3: To close the application inside browser just close the tab
# Step 4: To close the camera stop the application by exiting the shell or terminate the application(if using IDE)
# -------------------------------------------------------------------------------------------------------------------
