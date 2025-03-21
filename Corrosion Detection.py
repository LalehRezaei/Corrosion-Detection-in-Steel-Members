import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

# Primary Setting
st.set_page_config(page_title="Corrosion Detection", page_icon="🔧")

#Loading the Pre-trained Model
model = EfficientNetB0(weights='imagenet')  

# Prediction Function 
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  #Resizing the Image 
    img_array = image.img_to_array(img)  # Changing Image to an Array 
    img_array = np.expand_dims(img_array, axis=0)  #Adding Dim to an Array 
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  #Processing Input 

    #Make Prediction 
    prediction = model.predict(img_array)
    
    return prediction

#Function for Decoding the Prediction 
from tensorflow.keras.applications.efficientnet import decode_predictions
def get_class_prediction(prediction):
    decoded = decode_predictions(prediction, top=1)[0] #Decoding a Prediction to a Class Name  
    return decoded[0]  #Showing the Class and its Probability 

#ُStreamlit User Interface 
st.title('Corrosion Detection Web App')
st.write("This Web Application is Designed to Detect and Assess Corrosion in Steel Members. Please Upload One or More Images.")

#Uploading an Image 
uploaded_files = st.file_uploader("Choose Steel Member Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

#Folder for Saving Files 
temp_folder = './temp/'
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

#Processing and Showing Results 
if uploaded_files:
    for uploaded_file in uploaded_files:
        #Saving Image in a Temporary Path 
        img_path = os.path.join(temp_folder, uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        #Predicting Model 
        prediction = predict_image(img_path)
        
        #Decoding and Extracting Class and Severity 
        class_label, class_name, class_probability = get_class_prediction(prediction)

        #Displaying the Original Image and Prediction Result 
        st.image(uploaded_file, caption="Original Image", use_container_width=True)
        st.write(f"Predicted Class: {class_name}")
        st.write(f"Prediction Probability: {class_probability:.2f}")

        #Displaying Image using Image Processing Techniques 
        st.write("Display of Corroded Areas (Using Image Processing Techniqes)")

        #Thresholding for Detectiong Corroded Areas 
        img_opened = cv2.imread(img_path)
        gray = cv2.cvtColor(img_opened, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #Drawing the Corroded Areas on the Image 
        for contour in contours:
            if cv2.contourArea(contour) > 500: #Filtering Small Areas   
                cv2.drawContours(img_opened, [contour], -1, (0, 255, 0), 3)

        #Displaying the Processed Image 
        st.image(img_opened, caption="Processed Image (Corroded Areas Highlighted)", use_container_width=True)

#Running Application        
if __name__ == "__main__":
    st.write("The Web Application is Running...")
