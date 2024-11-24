# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st

# Load dataset and assign columns
columns = ['Sepal-length', 'Sepal-width', 'Petal-length', 'Petal-width', 'Class-label']
df = pd.read_csv("iris.data.csv", names=columns)

# Basic data analysis
print("First 5 rows of the dataset:")
print(df.head())

print("Last 5 rows of the dataset:")
print(df.tail())

print("Dataset summary:")
print(df.describe())

# Visualizing the dataset
sns.pairplot(df, hue='Class-label')
plt.show()

# Splitting the dataset into features and target
data = df.values
x = data[:, 0:4]  # Features
y = data[:, 4]  # Target

# Splitting data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)

# KNN Model
print("Training KNN Model...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
KNN_accuracy = accuracy_score(y_test, y_pred)
print("KNN Model Accuracy: " + str(KNN_accuracy * 100) + "%")

# Decision Tree Model
print("Training Decision Tree Model...")
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
y_pred = tree.predict(x_test)
tree_accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Model Accuracy: " + str(tree_accuracy * 100) + "%")

# SVM Model
print("Training SVM Model...")
svc = SVC()
svc.fit(x_train, y_train)
svm_pred = svc.predict(x_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("SVM Model Accuracy: " + str(svm_accuracy * 100) + "%")

# Plotting the comparison of models
algo = ["KNN", "Decision Tree", "SVM"]
accuracy_scores = [KNN_accuracy, tree_accuracy, svm_accuracy]
plt.title("Comparison of 3 Different models")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy Score")
plt.bar(algo, accuracy_scores, color=['blue', 'red', 'orange'])
plt.ylim(0, 1)
plt.show()

# Save the trained models using pickle
file_name1 = "KNN_save_model.sav"
file_name2 = "TREE_save_model.sav"
file_name3 = "SVM_save_model.sav"
pickle.dump(knn, open(file_name1, "wb"))
pickle.dump(tree, open(file_name2, "wb"))
pickle.dump(svc, open(file_name3, "wb"))

print("Models saved successfully!")

# Loading the models
knn_loaded = pickle.load(open(file_name1, "rb"))
tree_loaded = pickle.load(open(file_name2, "rb"))
svc_loaded = pickle.load(open(file_name3, "rb"))

print("Models loaded successfully!")

# Sample prediction
sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Example input
print("KNN Prediction:", knn_loaded.predict(sample_data))
print("Decision Tree Prediction:", tree_loaded.predict(sample_data))
print("SVM Prediction:", svc_loaded.predict(sample_data))

# Streamlit app for prediction
st.title("Iris Flower Classification")
st.write("Choose a machine learning model to classify iris flowers.")

# User input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Model selection
model_choice = st.selectbox(
    "Select a model",
    ["KNN", "Decision Tree", "SVM"]
)

# Prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    if model_choice == "KNN":
        prediction = knn_loaded.predict(input_data)
        st.write(f"Predicted Species (KNN): {prediction[0]}")
    elif model_choice == "Decision Tree":
        prediction = tree_loaded.predict(input_data)
        st.write(f"Predicted Species (Decision Tree): {prediction[0]}")
    elif model_choice == "SVM":
        prediction = svc_loaded.predict(input_data)
        st.write(f"Predicted Species (SVM): {prediction[0]}")

# Species mapping note
st.write("""
### Note:
- Species are mapped as follows:
  - 0: Setosa
  - 1: Versicolor
  - 2: Virginica
""")
