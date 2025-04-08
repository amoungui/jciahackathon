from tensorflow import keras

def create_cnn_model(input_size, num_classes, dropout_factor):
    """
    Crée un modèle CNN séquentiel pour la classification.

    Parameters:
    - input_size: Taille de l'image d'entrée (entier, ex. 32 pour 32x32 pixels).
    - num_classes: Nombre de classes de sortie.
    - dropout_factor: Taux de dropout (ex. 0.5 pour 50%).

    Returns:
    - model: Modèle Keras Sequential compilé.
    """
    model = keras.Sequential()

    # Feature extractor: extraction des caractéristiques
    model.add(keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(input_size, input_size, 3)))
    model.add(keras.layers.AveragePooling2D(2, 2))
    model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(keras.layers.AveragePooling2D(2, 2))
    model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(keras.layers.Dropout(dropout_factor))
    model.add(keras.layers.AveragePooling2D(2, 2))

    # Model adaptor: adaptation des caractéristiques
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))

    # Classifier head: classification finale
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax', name='classifier_head'))

    return model
