import os
import cv2
import numpy as np
import torch
from datetime import datetime
import csv
from tkinter import *
from PIL import Image, ImageTk
from facenet_pytorch import InceptionResnetV1, MTCNN

# Initialize face recognition models
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Attendance CSV setup
attendance_log_file = "attendance_log.csv"
marked_today = set()

if not os.path.exists(attendance_log_file):
    with open(attendance_log_file, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Timestamp"])

def log_attendance(name):
    if name not in marked_today:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(attendance_log_file, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, timestamp])
        marked_today.add(name)

def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            faces = []
            for box in boxes:
                face = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0)
                encoding = resnet(face_tensor).detach().numpy().flatten()
                faces.append(encoding)
            return faces, boxes
    return [], None

def encode_known_faces(known_faces):
    encodings = []
    names = []
    for name, path in known_faces.items():
        image = cv2.imread(path)
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encoding, _ = detect_and_encode(image_rgb)
            if encoding:
                encodings.append(encoding[0])
                names.append(name)
    return encodings, names

def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized = []
    for test in test_encodings:
        distances = np.linalg.norm(known_encodings - test, axis=1)
        idx = np.argmin(distances)
        if distances[idx] < threshold:
            recognized.append(known_names[idx])
        else:
            recognized.append("Not Recognized")
    return recognized

# Load known faces
known_faces = {
    "Sam Sudheer": "Images/sam.jpg",
    "Rhydhu V Ajith": "Images/Rhydhu.jpg",
    "Navneeth": "Images/Navneeth.jpg",
    "Siona": "Images/siona.jpg"
}
known_face_encodings, known_face_names = encode_known_faces(known_faces)
known_face_encodings = np.array(known_face_encodings)

# ---------------------- Tkinter GUI ---------------------- #
root = Tk()
root.title("Face Recognition Attendance")
root.geometry("800x600")

video_label = Label(root)
video_label.pack()

status_label = Label(root, text="Status: Waiting", font=("Helvetica", 14))
status_label.pack(pady=10)

cap = None

def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    show_frame()

def show_frame():
    global cap
    if cap is None:
        return
    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    test_encodings, boxes = detect_and_encode(frame_rgb)

    if test_encodings and known_face_encodings is not None:
        names = recognize_faces(known_face_encodings, known_face_names, test_encodings)
        for name, box in zip(names, boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = name
            if name != "Not Recognized":
                log_attendance(name)
                label += " - Marked"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            status_label.config(text=f"Status: {label}")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, show_frame)

def stop_camera():
    global cap
    if cap:
        cap.release()
        cap = None
        status_label.config(text="Status: Camera stopped")
        video_label.config(image="")

start_btn = Button(root, text="Start Camera", command=start_camera, font=("Helvetica", 12))
start_btn.pack(pady=5)

stop_btn = Button(root, text="Stop Camera", command=stop_camera, font=("Helvetica", 12))
stop_btn.pack(pady=5)

exit_btn = Button(root, text="Exit", command=root.destroy, font=("Helvetica", 12))
exit_btn.pack(pady=10)

root.mainloop()
