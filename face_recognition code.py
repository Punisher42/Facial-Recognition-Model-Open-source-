import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

vid_cap = cv2.VideoCapture(0)
my_img = face_recognition.load_image_file("your picture file/directory")
my_encoding = face_recognition.face_encodings(my_img)[0]

known_face_encodings = [my_encoding]
known_face_names = ["Your name "]
students = known_face_names.copy()

face_locations = []
face_encodings = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
a = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(a)

while True:
    _, frame = vid_cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_COMPLEX
            bottomLeftCornerOfText = (5, 70)
            fontScale = 1
            fontColor = (255, 0, 0)
            thickness = 2
            lineType = 3
            cv2.putText(frame, name + "Detected", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if name in students:
             students.remove(name)
             current_time =now.strftime("%H-%M-%S")
             lnwriter.writerow([name, current_time])


    cv2.imshow("Facial Detection", frame)
    if cv2.waitKey(1) & 0xFF ==ord("w"):
        break

vid_cap.release()
cv2.destroyAllWindows()
a.close()