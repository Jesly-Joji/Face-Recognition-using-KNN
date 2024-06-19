import numpy as np
import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
with open('data/faces.pkl', 'rb') as f:
    FACES = pickle.load(f)

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

# Scaling
sc = StandardScaler()
FACES = sc.fit_transform(FACES)

# Model initialization and training
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(FACES, LABELS)

# Load webcam
cap = cv2.VideoCapture(0)

# Face detection classifier
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = gray[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        
        # Standardize the face data
        resized_img = sc.transform(resized_img)
        
        # Predict the label
        output = knn.predict(resized_img)
        label = output[0]
        
        # Draw rectangle and label on the image
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    cv2.imshow("Face Recognition", img)
    
    # Terminate the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
