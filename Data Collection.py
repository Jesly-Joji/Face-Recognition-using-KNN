import cv2
import numpy as np
import pickle
import os

# Create the data folder if it doesn't exist
if not os.path.exists("data"):
    os.mkdir("data")

# Load webcam
cap = cv2.VideoCapture(0)  # 0 -> primary webcam

# Load face detection classifier
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Collecting data parameters
total = 100
face_data = []
labels = []

# Input user name
name = input("Enter your Name: ")


count = 0  # Initialize counter
while True:
    # Read image frame
    ret, img = cap.read()

    if not ret:
        continue

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = gray[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, str(count+1), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        face_data.append(resized_img)
        count += 1

    cv2.imshow("Data Collection", img)

    # Termination condition
    if count >= total or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert to numpy array
face_data = np.array(face_data)
face_data = face_data.reshape(total, -1)
print(face_data.shape)
print(face_data.ndim)
print(face_data)


#SAVE FACE AND LABELS DATA

if 'names.pkl' not in os.listdir('data/'):
    names = [name]*total
    with open('data/names.pkl', 'wb') as file:
        pickle.dump(names, file)
else:
    with open('data/names.pkl', 'rb') as file:
        names = pickle.load(file)

    names = names + [name]*total
    with open('data/names.pkl', 'wb') as file:
        pickle.dump(names, file)


if 'faces.pkl' not in os.listdir('data/'):
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(face_data, w)
else:
    with open('data/faces.pkl', 'rb') as w:
        faces = pickle.load(w)

    faces = np.append(faces, face_data, axis=0)
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(faces, w)


