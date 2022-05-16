import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from tensorflow import keras

model = keras.models.load_model("mobilnet_model.h5")
dog_breed  = pd.read_csv('dog_breed.csv', delimiter=',').columns.tolist()
 

st.title("Application streamlit pour déterminer la race d'un chien")
st.image(Image.open("snooky.jpg"), caption='Snooky le chien')

with st.container():
    st.write(
        "Le modèle de réseau de neurones utilisé est disponible sous image en téléchargement.")
    st.write("Le loulouououououopu modèle de réseau de neurones est basé sur un mobilenet à été entraîné avec un dataset de  300 photos de chiens (100 photos de chihuahua 100 photos de labrador et 100 photos de husky). Le nombre d'epoque est de 20, le nombre de batch est de 16 et le learning rate est de 0.0001. ")
    st.write("")
    st.image(Image.open("fonction_exactitude_mobilnet.png"), caption='Fonction exactitude du réseau de neurone basé sur mobilnet')
    st.image(Image.open("fonction_perte_mobilnet.png"), caption='Fonction perte du réseau de neurone basé sur mobilnet')
    st.write("")
    with open("schema_model_cnn.jpg", "rb") as file:
        btn = st.download_button(
            label="Télécharger le modèle",
            data=file,
            file_name="schema_model_cnn.jpg",
            mime="image/jpg")
    

st.write("")
st.write("")
st.write("")
st.write("")

with st.container():
    st.write("Veuillez insérer une image de chien")
    uploaded_file = st.file_uploader("Insérer l'image")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Voici image uploader')
        st.write("")
        st.write("La taille de l'image initiale :")
        st.write(img.size)
        img = img.resize((128, 128))
        img_array = np.array(img)
        #img_array = img_array/255
        img_array = np.expand_dims(img_array, axis=0)
        st.write(
            "La taille de la matrice associée à l'image après le prétraitement de l'image :")
        st.write(img_array.shape)
        result = model.predict(img_array)
        st.write("Voici le résultat de prédiction en fonction des catégories :")
        st.write(result)
        result_cat = result.argmax(axis=-1)
        if result_cat == 0:
            result_cat = dog_breed[0]
        elif result_cat == 1:
            result_cat = dog_breed[1]
        elif result_cat == 2:
            result_cat = dog_breed[2]

        st.write("Le modèle détecte que la race de chien présente sur la photo est :")
        st.write(result_cat)