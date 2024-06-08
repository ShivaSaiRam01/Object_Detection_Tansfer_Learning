# VGG16

import streamlit as st
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.preprocessing import image
import numpy as np

# Initialize the VGG16 model
model = tf.keras.applications.VGG16()

def main():
    st.title('Class identification in Image VGG16')
    
    # File uploader for the image
    img_path = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if img_path is not None:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)

        # Identify the object when the button is clicked
        if st.button('Predict'):
            predictions = model.predict(img_array_expanded_dims)
            results = imagenet_utils.decode_predictions(predictions)
            st.write('The Class is:', results[0][0][1])

if __name__ == '__main__':
    main()