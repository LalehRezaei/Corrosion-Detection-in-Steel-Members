import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Corrosion Detection", page_icon="🔧")

model = EfficientNetB0(weights='imagenet')  

def predict_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  

    prediction = model.predict(img_array)
    
    return prediction

from tensorflow.keras.applications.efficientnet import decode_predictions
def get_class_prediction(prediction):
    decoded = decode_predictions(prediction, top=1)[0]  
    return decoded[0]  

st.title('')
st.write("")

uploaded_files = st.file_uploader("image", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

temp_folder = './temp/'
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

if uploaded_files:
    for uploaded_file in uploaded_files:

        img_path = os.path.join(temp_folder, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        prediction = predict_image(img_path)
        
        class_label, class_name, class_probability = get_class_prediction(prediction)

        st.image(uploaded_file, caption="", use_column_width=True)
        st.write(f" {class_name}")
        st.write(f"{class_probability:.2f}")

        st.write("")

        img_opened = cv2.imread(img_path)
        gray = cv2.cvtColor(img_opened, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  
                cv2.drawContours(img_opened, [contour], -1, (0, 255, 0), 3)

        st.image(img_opened, caption="", use_column_width=True)
        
if __name__ == "__main__":
    st.write("")
