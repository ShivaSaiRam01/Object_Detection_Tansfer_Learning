# import streamlit as st
# import tensorflow as tf
# st.echo("Importing imgnetutils...")
# from keras.applications import imagenet_utils
# st.echo("Import successful!")
# # from keras .preprocessing.image import ImageDataGenerator
# st.echo("Importing image...")
# from keras.preprocessing import image
# st.echo("Import successful!")
# import numpy as np
# # import keras


# # model = keras.applications.ResNet152()
# st.echo("Importing resnet152...")
# model = tf.keras.applications.ResNet152()
# st.echo("Import successful!")

# # class_names_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
# # with open (class_names_path, 'r') as f:
# #   class_names = f.read().splitlines()
# # ===========================================

# def main():

#     st.title('Object identification in the image')
#     img_path = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


#     img = image.load_img(img_path ,target_size = (224,224))

#     img_array = image.img_to_array(img)
#     img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#     predictions = model.predict(img_array_expanded_dims)
#     results = imagenet_utils.decode_predictions(predictions)



#     if st.button('Identify The object'):
#         st.write(results[0][0][1])


# if __name__=='__main__':
#     main()
# #===============================================================================


import streamlit as st
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.preprocessing import image
import numpy as np

# Initialize the ResNet152 model
model = tf.keras.applications.ResNet152()

def main():
    st.title('Object identification in the image')
    
    # File uploader for the image
    img_path = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if img_path is not None:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)

        # Identify the object when the button is clicked
        if st.button('Animal identification in Image ResNet152'):
            predictions = model.predict(img_array_expanded_dims)
            results = imagenet_utils.decode_predictions(predictions)
            st.write('The Animal is:', results[0][0][1])

if __name__ == '__main__':
    main()
