import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from typing import List

def get_model(
    base_model, 
    dense_units: List[int], 
    num_classes: int, 
    dropout: float,
    freeze_layers: bool = True
) -> tf.keras.Model:

    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False
            
    target_size = base_model.input_shape[1:]
    model = Sequential(
        [
            layers.Input(target_size),
            base_model
        ]
    )

    for units in dense_units:
        if units != 0:
            model.add(layers.Dense(units, activation='relu'))
            if dropout > 0:
                model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model