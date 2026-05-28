import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# data
iris = load_iris()
X, y = iris.data, iris.target

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model ANN
model = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# ================= UI =================
st.title("Iris Flower Classification")
st.write("Simple machine learning project for flower prediction")

# input
st.sidebar.header("Input Data")

sl = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.1)
sw = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5)
pl = st.sidebar.slider("Petal Length", 1.0, 7.0, 1.4)
pw = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.2)

data = scaler.transform([[sl, sw, pl, pw]])

# prediksi
if st.button("Predict"):
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0]

    st.subheader("Result")
    st.write("Prediction:", iris.target_names[pred])

    st.write("Probability")
    for i in range(3):
        st.write(f"{iris.target_names[i]} : {prob[i]*100:.2f}%")

# akurasi
st.subheader("Model Accuracy")
st.write(f"{acc*100:.2f}%")

# confusion matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")

for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center")

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# visualisasi data
st.subheader("Data Visualization")

fig2, ax2 = plt.subplots()

scatter = ax2.scatter(X[:, 0], X[:, 2], c=y, cmap="viridis")

ax2.set_xlabel("Sepal Length")
ax2.set_ylabel("Petal Length")

ax2.legend(*scatter.legend_elements(), title="Class")

st.pyplot(fig2)