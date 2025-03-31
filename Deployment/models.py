import tensorflow as tf
from tensorflow import keras


IMG_SIZE = 254

# Define metrics
METRICS = [
      keras.metrics.F1Score(name='f1_score',threshold=(0.5)),
      keras.metrics.AUC(name='auc'),
]

def make_model_binary(summarise='False'):
    '''
    Builds and compiles a CNN model for binary classification.

    Inputs:
      summarise: bool, whether to print the model summary (default is False).

    Returns:
      model: Compiled Keras model.
    '''
    model = keras.Sequential([
      keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
      keras.layers.Conv2D(32, 3, padding='same'),
      keras.layers.LeakyReLU(),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPooling2D(),
      keras.layers.Conv2D(64, 3, padding='same'),
      keras.layers.LeakyReLU(),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPooling2D(),
      keras.layers.Conv2D(64, 3, padding='same'),
      keras.layers.LeakyReLU(),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPooling2D(),
      keras.layers.Conv2D(32, 3, padding='same'),
      keras.layers.LeakyReLU(),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dense(32),
      keras.layers.LeakyReLU(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)

    if summarise == 1:
      model.summary()
      return model
    else:
      return model

def make_model_ternary(summarise=False):
    '''
    Build and compile a CNN model for multi-class classification.


    Inputs:
     summarise: str, if '1', it prints the model summary. Default is 'False'.

    Returns:
     model: Compiled Keras model ready for training.
    '''

    model = keras.Sequential([
        keras.layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        keras.layers.Conv2D(32, 3, padding='same'),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same'),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same'),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same'),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(32),
        keras.layers.LeakyReLU(alpha=0.1),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    if summarise == 1:
        model.summary()
        return model
    else:
        return model
