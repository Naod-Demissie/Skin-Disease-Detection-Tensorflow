
import tensorflow as tf
from tensorflow.keras.applications import (EfficientNetV2B0, EfficientNetV2B1, 
                                           EfficientNetV2B2, EfficientNetV2B3, 
                                           EfficientNetV2S, EfficientNetV2M)


BATCH_SIZE = 24
NUM_CLASSES = 3
NUM_EPOCHS = 50

# paths
ROOT_DIR = os.getcwd()
# DATA_DIR = 
TB_LOG_DIR  = f'D:/Projects/Skin-Disease-Detection-Tensorflow/logs/tensorboard'
TUNER_DIR = f'{TB_LOG_DIR}/tuner'

CSV_LOG_DIR  = f'D:/Projects/Skin-Disease-Detection-Tensorflow/logs/'
CKPT_DIR  = f'D:/Projects/Skin-Disease-Detection-Tensorflow/checkpoints/'


TARGET_SIZE = {
    'efficientnetv2-b0': (224, 224, 3), 
    'efficientnetv2-b1': (240, 240, 3), 
    'efficientnetv2-b2': (260, 260, 3), 
    'efficientnetv2-b3': (300, 300, 3), 
    'efficientnetv2-s': (384, 384, 3), 
    'efficientnetv2-m': (480, 480, 3), 
}

def get_base_model(base_model_name):
    assert base_model_name in [
        'efficientnetv2-b0', 'efficientnetv2-b1', 
        'efficientnetv2-b2', 'efficientnetv2-b3', 
        'efficientnetv2-s', 'efficientnetv2-m', 
    ], f'Invalid base model name: {base_model_name}'

    return (
        EfficientNetV2B0(include_top=False, input_shape=(224, 224, 3), pooling='avg') if base_model_name == 'efficientnetv2-b0' else
        EfficientNetV2B1(include_top=False, input_shape=(240, 240, 3), pooling='avg') if base_model_name == 'efficientnetv2-b1' else
        EfficientNetV2B2(include_top=False, input_shape=(260, 260, 3), pooling='avg') if base_model_name == 'efficientnetv2-b2' else
        EfficientNetV2B3(include_top=False, input_shape=(300, 300, 3), pooling='avg') if base_model_name == 'efficientnetv2-b3' else
        EfficientNetV2S(include_top=False, input_shape=(384, 384, 3), pooling='avg') if base_model_name == 'efficientnetv2-s' else
        EfficientNetV2M(include_top=False, input_shape=(480, 480, 3), pooling='avg')
    )










