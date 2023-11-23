
import tensorflow as tf
from tensorflow.keras.applications import (EfficientNetV2B0, EfficientNetV2B1, 
                                           EfficientNetV2B2, EfficientNetV2B3, 
                                           EfficientNetV2S, EfficientNetV2M)


BATCH_SIZE = 16
NUM_CLASSES = 3
NUM_EPOCHS = 20

# paths
ROOT_DIR = "C:/Users/AII/Documents/Naod-Documents/Skin-Disease-Detection-Tensorflow"
DATA_DIR = 'C:/Users/AII/Documents/Olyad/test/Skin-Disease-Detection-Pytorch/data/raw'

CSV_LOG_DIR  = f'{ROOT_DIR}/logs/csv_logs'
TB_LOG_DIR  = f'{ROOT_DIR}/logs/tensorboard'
TUNER_DIR = f'{TB_LOG_DIR}/tuner'
CKPT_DIR  = f'{ROOT_DIR}/checkpoints/'


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


