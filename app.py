import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

# Load the trained model
model = load_model("brain-tumors.h5")

def predict_image(image):
    imag = image.resize((64, 64))
    img = np.array(imag)
    input_img = np.expand_dims(img, axis=0)

    # Make prediction using the model
    result = model.predict(input_img)

    return result

def main():
    st.title("Brain Tumor Prediction App")

    uploaded_file = st.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("predict"):
        # Make prediction
            result = predict_image(image)

        # Display prediction result
            if result[0][0]==1.:
               st.success("Prediction: The image has a brain tumor.")
            else:
               st.success("Prediction: The image does not have a brain tumor.")

if __name__ == "__main__":
    main()
