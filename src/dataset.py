import sys
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import math

from glob import glob

from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.utils import Sequence, to_categorical

from typing import List, Tuple

sys.path.append('..')
from sys.config import BATCH_SIZE, TARGET_SIZE, NUM_CLASSES, DATA_DIR


# Custom Data Generator
class ImageDataGenerator(Sequence):
    def __init__(
            self, 
            df: pd.DataFrame, 
            batch_size: int, 
            targe_size: Tuple[int], 
            n_classes: int, 
            shuffle: bool =True, 
            training: bool =True
    ) -> None:
        
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.targe_size = targe_size[:-1]
        self.n_classes = n_classes
        self.training = training
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle and self.training:
            np.random.shuffle(self.indexes)
            return self.df.reindex(self.indexes)

    def __getitem__(self, index):
        indexes_ = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        batch_imgs = []
        for idx in indexes_:
            img_path = self.df.loc[idx, 'img_path']
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            xmin, ymin, xmax, ymax = self.df.loc[idx, ['xmin', 'ymin', 'xmax', 'ymax']].values
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            img = img[ymin:ymax, xmin:xmax]
            img = cv2.resize(img, self.targe_size, interpolation=cv2.INTER_LINEAR)

            # Image augmentation
            if self.training:
                transform = A.Compose(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5)
                    ]
                )
                augments = transform(image=img)
                img = augments["image"]
            batch_imgs.append(img)
        return np.array(batch_imgs), to_categorical(self.df.loc[indexes_, 'sparse_label'], num_classes=self.n_classes)
    


# Data Preprocessing
df = pd.read_csv(f'{DATA_DIR}/verified_annotation_from_xml.csv')
df['img_path'] =f'{DATA_DIR}/images/' + df['image_name']
df.drop(columns=['Unnamed: 0'], inplace=True)
df['label_name'] = df['label_name'].apply(lambda x: x.lower())
df['sparse_label'] = df['label_name'].map({'atopic':0,'papular':1,'scabies':2})


# Group shuffling with the patient id
gs = GroupShuffleSplit(n_splits=2, train_size=.85, random_state=42)
train_val_idx, test_idx = next(gs.split(df,groups=df.patient_id))
train_val_df = df.iloc[train_val_idx]
test_df = df.iloc[test_idx]

train_idx, val_idx = next(gs.split(train_val_df, groups=train_val_df.patient_id))
train_df = train_val_df.iloc[train_idx]
val_df = train_val_df.iloc[val_idx]

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


train_generator = ImageDataGenerator(
    df=train_df, 
    batch_size=BATCH_SIZE, 
    targe_size=TARGET_SIZE, 
    n_classes=NUM_CLASSES, 
    shuffle=True, 
    training=True
)
val_generator = ImageDataGenerator(
    df=val_df, 
    batch_size=BATCH_SIZE, 
    targe_size=TARGET_SIZE, 
    n_classes=NUM_CLASSES, 
    shuffle=True, 
    training=True
)
test_generator = ImageDataGenerator(
    df=test_df, 
    batch_size=BATCH_SIZE, 
    targe_size=TARGET_SIZE, 
    n_classes=NUM_CLASSES, 
    shuffle=False, 
    training=False
)


