# ProjetSnookyDeepLearning
<!-- ABOUT THE PROJECT -->
## A propos du projet

Dépôt pour le projet SnookyDeepLearning  réalisé par Victorien, Eric, Marceline et Alexia.

<p align="right">(<a href="#top">back to top</a>)</p>



### Projet réalisé avec :

Python et les librairies suivantes
* Streamlit
* Pandas
* Numpy
* Keras
* Tensorflow
* PIL
* os

Docker pour le déploiment de l'application streamlit

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Installation de l'environnement de travail

### Prérequis

* Avoir un ide compatible Python
* Avoir Docker (optionnel)

### Installation

1. Cloner le dépôt
2. Créer un environnement virtuel nommé venv dans le dépôt clôné localement
3. Installé les paquets Python du requirements.txt dans l'environnement virtuel
4. Mettre le dossier dataset_dog dans le dépôt clôné à la racine

Pour l'application dans Docker

* Exécuter le scripte bash CreationConteneurApplicationStreamlit.sh pour créé le conteneur Docker nommé app_prediction_chien
* Pour lancer le conteneur app_prediction_chien exécuter le scripte LancementApplicationStreamlitDocker (fonctionne uniquement sur Linux)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Algorithme basé sur un réseau de neurone de type réseau de neurone convolutif qui prédit la race d'un chien.

<p align="right">(<a href="#top">back to top</a>)</p>
