import os
import sys

import keras_tuner
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard

# sys.path.append(os.getcwd())
# from src.config import *
# from src.dataset import ImageDataGenerator, train_df, val_df

sys.path.append(os.pardir)
from src.config import *
from src.dataset import ImageDataGenerator, train_df, val_df

def buildModel(hp):
    base_model_names = [
        'efficientnetv2-b0', 'efficientnetv2-b1', 
        'efficientnetv2-b2', 
        # 'efficientnetv2-b3', 
        # 'efficientnetv2-s', 
        # 'efficientnetv2-m', 
    ]
    base_model_name = hp.Choice('base_model', base_model_names)
    base_model = get_base_model(base_model_name)
    x = base_model.output

    if hp.Boolean('batch_norm(base)'):
        x = layers.BatchNormalization()(x)
    
    num_layers = hp.Choice('num_layers', [1, 2, 3])
    for idx, _ in enumerate(range(1, num_layers+1)):
        units= hp.Int(f'dense_units({idx})', min_value=64, max_value=128, step=32) 
        x = layers.Dense(units=units, activation='relu')(x)
        if hp.Boolean(f'batch_norm({idx})'):
            x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.Float(f'dropout({idx})', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    lr = hp.Float('init_lr', min_value=0.0001, max_value=0.001, step=0.0002)  
    model.compile(
        loss='categorical_crossentropy',
        optimizer= Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    return model

if __name__ == '__main__':
    
    base_model_name = 'efficientnetv2-b2' 
    train_generator = ImageDataGenerator(
        df=train_df, 
        batch_size=BATCH_SIZE, 
        base_model_name=base_model_name, 
        n_classes=NUM_CLASSES, 
        shuffle=True, 
        training=True
    )
    val_generator = ImageDataGenerator(
        df=val_df, 
        batch_size=BATCH_SIZE, 
        base_model_name=base_model_name, 
        n_classes=NUM_CLASSES, 
        shuffle=True, 
        training=True
    )
    hp = keras_tuner.HyperParameters()
    tuner = keras_tuner.RandomSearch(
        hypermodel= buildModel,
        objective='val_accuracy',
        max_trials=1000,
        executions_per_trial=1,
        directory=f'{ROOT_DIR}/logs',
        project_name='search_trails'
    )

    tuner.search(
        train_generator,
        validation_data=val_generator,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[TensorBoard(log_dir=TUNER_DIR)]
    )

