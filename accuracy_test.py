import numpy as np
import cv2
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
with open('data/faces.pkl', 'rb') as f:
    FACES = pickle.load(f)

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(FACES, LABELS, test_size=0.3, random_state=42)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model initialization and training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

# Prediction
pred = knn.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(Y_test, pred)
precision = precision_score(Y_test, pred, average='weighted')  # Use average='weighted' for multi-class
recall = recall_score(Y_test, pred, average='weighted')  # Use average='weighted' for multi-class

# Print scores
print("Accuracy",accuracy*100)
print("Precision",precision*100)
print("Recall:",recall*100)

# Confusion Matrix
cm = confusion_matrix(Y_test, pred)

# Plotting the Confusion Matrix
plt.figure(figsize=(10, 7))
plot=sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(LABELS), yticklabels=np.unique(LABELS))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


plot.figure.savefig("confusion_matrix.png")
