#!/bin/bash

#Construction de l'image streamlit_prediction_chien en version 1.0 à partir du Dockerfile
echo 'Construction de  image docker streamlit_prediction_chien'
docker build -t streamlit_prediction_chien:1.0 .
echo 'Construction de  image docker streamlit_prediction_chien terminée'
#Création du conteneur app_prediction_chien à partir de l'image streamlit_prediction_chien
echo 'Construction du  conteneur app_prediction_chien'
docker run --name app_prediction_chien -it streamlit_prediction_chien:1.0
echo 'Construction du  conteneur app_prediction_chien terminée'