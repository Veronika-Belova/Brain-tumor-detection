import streamlit as st
import torch
import requests
from PIL import Image
from io import BytesIO
import time
import numpy as np
import os

@st.cache_data()
def load_model(model_path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

def get_prediction(model, image):
    start_time = time.time()
    output = model(image)
    output_img = output.render()[0]
    end_time = time.time()
    response_time = round(end_time - start_time, 4)
    st.image(output_img)
    st.write("Response Time:", response_time, "seconds")

def is_image_file(file):
    return file.type.startswith('image/')

model_options = ["About the project", "Model 1", "Model 2", "Model 3"]
selected_model = st.sidebar.radio("Options", model_options)

if selected_model == "About the project":
  st.markdown("<h1 style='text-align: center;'>Brain tumor object detection</h1>", unsafe_allow_html=True)
  st.image('data/Вступление.jpeg', use_column_width=True)
  st.markdown("<h3 style='font-size: 28px;'>Задача проекта:</h3>", unsafe_allow_html=True)
  st.markdown("""
  #### Детекция объектов с использованием YOLOv5
  Модель обучалась на трех датасетах: Axial, Coronal, Sagittal
  #### Состав датасета Axial:
  - Train: 310 files
  - Test: 75 files
  #### Состав датасета Coronal:
  - Train: 319 files
  - Test: 78 files
  #### Состав датасета Sagittal:
  - Train: 264 files
  - Test: 70 files
  """)
  st.image('data/Вступление.png', use_column_width=True)
  st.markdown("<h3 style='font-size: 24px;'>Разделение на плоскости: осевая, корональная и сагиттальная</h3>", unsafe_allow_html=True)

elif selected_model in ["Model 1", "Model 2", "Model 3"]:
    if selected_model == "Model 1":
        model_path = 'best.pt'
        st.markdown("<h1 style='text-align: center;'>Axial</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 18px;'>Train - 60 Epoch, Batch size - 12 </h3>", unsafe_allow_html=True)
        st.image('data/axial_batch_pred.jpg', use_column_width=True)
        st.image('data/confusion_matrix_axial.png', use_column_width=True)
        st.image('data/PR_curve_axial.png', use_column_width=True)
        st.image('data/results_axial.png', use_column_width=True)
    elif selected_model == "Model 2":
        model_path = 'best_2.pt'
        st.markdown("<h1 style='text-align: center;'>Coronal</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 18px;'>Train - 60 Epoch, Batch size - 12 </h3>", unsafe_allow_html=True)
        st.image('data/coronal_batch.jpg', use_column_width=True)
        st.image('data/matrix_coronal.png', use_column_width=True)
        st.image('data/PR_coronal.png', use_column_width=True)
        st.image('data/results_coronal.png', use_column_width=True)
    elif selected_model == "Model 3":
        model_path = 'best_3.pt'
        st.markdown("<h1 style='text-align: center;'>Sagittal</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 18px;'>Train - 60 Epoch, Batch size - 12 </h3>", unsafe_allow_html=True)
        st.image('data/sagittal_batch.jpg', use_column_width=True)
        st.image('data/matrix_sagittal.png', use_column_width=True)
        st.image('data/PR_sagittal.png', use_column_width=True)
        st.image('data/results_sagittal.png', use_column_width=True)

    model = load_model(model_path)
    model.conf = 0.6
    model.eval()

    st.title("Upload images for detection")
    url = st.text_input("Enter image url")

    uploaded_files = st.file_uploader("Upload multiple images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            try:
                if is_image_file(uploaded_file):
                    image = Image.open(uploaded_file).convert('RGB')
                    get_prediction(model, image)
                else:
                    st.warning("Invalid image format. Please upload a JPEG, JPG, or PNG image.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            
        if url:
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                get_prediction(model, image)
            except requests.exceptions.MissingSchema:
                st.warning("Invalid URL format. Make sure to include 'http://' or 'https://'")

