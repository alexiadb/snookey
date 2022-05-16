import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import random

from PIL import Image
from tensorflow.keras import utils



def dataset_preprocessing(base_dir, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, COLOR_DEPTH ):
    """
    Préparation des données à donner au réseau de neurones

    Args:
        base_dir (str): Arborescence du dossier qui contient le dataset
        BATCH_SIZE (int): Nombre d'image dans un batch
        IMG_HEIGHT (int): Hauteur de l'image en pixel
        IMG_WIDTH (int): Largeur de l'image en pixel
        COLOR_DEPTH (int): Nombre de couleur en rgb

    Return:
        train_ds (tensorflow.python.data.ops.dataset_ops.BatchDataset): Dataset d'entraînement
        val_ds (tensorflow.python.data.ops.dataset_ops.BatchDataset): Dataset de validation
        class_names(list): Liste des catégories que le réseau de neurone doit trouver
    """

    print("Création du dataset d'entraînement")
    train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir,'train'),
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT,IMG_WIDTH ),
    batch_size=BATCH_SIZE)
    print("")
    print("Création du dataset de validation")
    val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir,'train'),
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT,IMG_WIDTH ),
    batch_size=BATCH_SIZE)
    print("")
    print("Création du dataset de test")
    test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir,'test'),
    seed=123,
    image_size=(IMG_HEIGHT,IMG_WIDTH ),
    batch_size=BATCH_SIZE)
    print("")
    class_names = train_ds.class_names
    print('Les classes à déterminer : ',class_names)
    print("")
    print("Affichage des 5 première images du dataset d'entraînement")
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(5):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    print("")
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break
    print("image_batch est un  tensor de forme {}. Il s'agit  d'un batch de {} images de forme {}x{}x{}(la dernière dimension correspond à la colorisation RGB). Le label_batch est un tensor de forme {}, Il correspond au labels des {} images".format(image_batch.shape,image_batch.shape[0],image_batch.shape[1],image_batch.shape[2],image_batch.shape[3],labels_batch.shape, labels_batch.shape[0]))

    return(train_ds, val_ds, test_ds, class_names)




def model_presentation(model, train_ds, val_ds, number_epoch, optimizer, loss, metrics):
    """Entraîne le modèle de réseau de neuronne souhaité avec les hyperparamètres voullus et retourne les graphes de performance

    Args:
        model (keras.engine.functional.Functional): Modèle du reseau de neurone
        train_ds (tensorflow.python.data.ops.dataset_ops.BatchDataset): Dataset qui contient les images d'entraînement
        val_ds (tensorflow.python.data.ops.dataset_ops.BatchDataset): Dataset qui contient les images de validation
        number_epoch (int): Nombre d'époque du reseau de neurone
        optimizer (str (name of optimizer) ): nom de l'optimisateur
        loss (module): fonction de perte
        metrics (list): Liste des métriques pour évaluer le modèle

    """
    
    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
    model.summary() 
    utils.plot_model(
    model,
    to_file='schema_model_cnn.jpg',
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=True,
    dpi=96,
    layer_range=None,
    show_layer_activations=True
    )

    history = model.fit(
    train_ds, validation_data=val_ds,
    epochs=number_epoch
    )
    
    #Tracé de la perte pendant l'entrainement du modèle
    loss_value = history.history['loss']
    val_loss_value = history.history['val_loss']
    epochs = range(1, number_epoch+1) 
    plt.plot(epochs, loss_value, 'bo', label = 'Entrainement')
    plt.plot(epochs, val_loss_value, 'b', label = 'Validation')
    plt.xlabel("Nombre d\'époques")
    plt.ylabel("Perte")
    plt.title('Perte (loss) pendant l\'entrainement et la validation')
    plt.legend()
    plt.show()

    #Tracé de l'exactitude pendant l'entrainement du modèle
    acc_value = history.history['accuracy']
    val_acc_value = history.history['val_accuracy']
    epochs = range(1, number_epoch+1)
    plt.plot(epochs, acc_value, 'bo', label = 'Entrainement')
    plt.plot(epochs, val_acc_value, 'b', label = 'Validation')
    plt.xlabel("Nombre d\'époques")
    plt.ylabel("Exactitude de prédiction")
    plt.title('Exactitude (accuracy) pendant l\'entrainement et la validation')
    plt.legend()
    plt.show()


def prediction_dog_race(model, base_dir, class_names,  IMG_HEIGHT, IMG_WIDTH):
    """Prédit la race d'un chien à partir d"un modèle entraîné

    Args:
        model (keras.engine.functional.Functional): _description_
        base_dir (str): Arborescence du dossier qui contient le dataset
        class_names (list): Liste des catégories que le réseau de neurone doit trouver
        IMG_HEIGHT (int): Hauteur de l'image en pixel
        IMG_WIDTH (int): Largeur de l'image en pixel
    """

    path_directory = os.path.join(base_dir,'test')
    dog_race_random = random.choice(os.listdir(path_directory))
    path_directory_race = path_directory+'/'+dog_race_random
    picture_random = random.choice(os.listdir(path_directory_race))
    path_image = path_directory_race+'/'+picture_random
    img = Image.open(path_image)
    display(img)
    img_array = np.array(img.resize((IMG_HEIGHT,IMG_WIDTH)))
    img_array.shape
    img_array
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array.shape)

    preds = model.predict(img_array)
    print(preds)
    print(class_names [np.argmax(preds)])
    print("")
    print("La race de chien est:{} \nL'ordinateur a prédit : {}".format(dog_race_random,class_names [np.argmax(preds)]))