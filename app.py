import streamlit as st 
import pickle
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
#import cv2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input





# Set the app title 
st.title('Image Matching App') 
# Add a welcome message 
st.write('Please uplaod a test image!') 
# Create a text input 
# widgetuser_input = st.text_input('Enter a custom message:', 'Hello, Streamlit!') 
# Display the customized message 
# st.write('Customized Message:', widgetuser_input)


with st.sidebar:
    # st.[element_name]

    test_image = st.file_uploader("Choose a file")
    if test_image is not None:
        # To read file as bytes:
        # bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        st.image(test_image,width = 200, caption='Test Image')




def load_features_from_file(file_path):
    # Load image paths and features from a file using pickle
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data['image_paths'],loaded_data['class_labels'], loaded_data['features']

loaded_image_paths,loaded_class_labels, loaded_database_features = load_features_from_file('database_features_ann.pkl')
# st.write(loaded_image_paths)

# st.write('loaded features')

def find_similar_images(query_features, database_features,class_labels, image_paths, top_k=10):
    # Calculate cosine similarity between query and database images
    similarities = cosine_similarity([query_features], database_features)[0]
    # st.write(similarities)
    
    # Get indices of top-k similar images
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    # st.write(top_indices)
    dicti={}
    for m,x in enumerate(top_indices):
        dicti[class_labels[x]]=0

    for a,index in enumerate(top_indices):
        if dicti[class_labels[index]]==0:
            for b,j in enumerate(top_indices):
                if class_labels[index]==class_labels[j]:
                    dicti[class_labels[j]]+=similarities[j]
    sorted_dict = sorted(dicti.items(), key=lambda x:x[1], reverse=True)
    dicti = dict(sorted_dict)
    for keys in dicti.keys():
        st.write("class:",keys,"similarity:",dicti[keys])
        st.image(f"Database/{keys}/image_3.jpg",width=200)
    # st.write(dicti)
    # Display the results
    # print(f"Top {top_k} similar images:")
    # for i, index in enumerate(top_indices):
        # print(f"{i+1}. Image: {image_paths[index]}, Similarity: {similarities[index]}")


def preprocess_image(image_path):
    # Load and preprocess an image for EfficientNet
    # img = image.load_img(image_path, target_size=(224, 224))
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


## for test image
def extract_features(model, image_path):
    # Extract features using the EfficientNet model
    img_array = preprocess_image(image_path)
    features = model.predict(img_array)
    return features.flatten()


# def preprocess_image_uploaded_img(image_path):
#     # Load and preprocess an image for EfficientNet
#     img = image.load_img(test_image, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
#     return img_array

# def extract_features_uploaded_img(model, image_path):
#     # Extract features using the EfficientNet model
#     img_array = preprocess_image(image_path)
#     features = model.predict(img_array)
#     return features.flatten()



if test_image is not None:
    model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    # query_image_path = 'query/img_test1.jpeg'
    query_image_path = test_image



    query_features = extract_features(model, query_image_path)
    # st.write(query_features)

    find_similar_images(query_features, loaded_database_features, loaded_class_labels, loaded_image_paths, top_k=5)
    # st.write(output)
else :
    st.write('waiting for test image....')