#!/bin/bash

#Lancement du conteneur de l'application qui prédit la race de chien
echo ' D�marrage du  conteneur app_prediction_chien'
docker start app_prediction_chien
#Ouverture de l'application sur un navigateur web
xdg-open http://172.17.0.2:8502
docker exec  -it app_prediction_chien  streamlit run app.py
