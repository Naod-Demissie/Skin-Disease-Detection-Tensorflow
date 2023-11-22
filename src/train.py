import tensorflow as tf
from glob import glob

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scikitplot.metrics import plot_confusion_matrix

from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                        ReduceLROnPlateau, TensorBoard)

print(tf.__version__)


from typing import List, Tuple

def training_config(
    log_name: str,
    base_model: tf.keras.models.Model,
    dense_units: List[int],
    target_size: Tuple[int, int, int],
    n_classes: int,
    n_epochs: int = 20,
    init_lr: float = 0.0003,
    batch_size: int = 32,

    #TODO type annotation with optional
    class_weight=None,
    weight_path = None,
    monitor:str ='val_accuracy', 
) -> None:


    # callbacks
    checkpoint  = ModelCheckpoint(
        f'{CKPT_DIR}/{log_name}', 
        monitor=monitor, 
        verbose=1, 
        save_best_only=True, 
        save_freq="epoch"
    )
    tensorboard = TensorBoard(
        log_dir=f'{LOG_DIR}/{log_name}', histogram_freq=1
    )
    reduce_lr   = ReduceLROnPlateau(
        monitor=monitor, 
        factor=0.85, 
        patience=2, 
        verbose=1, 
        min_delta=0.0001
    )
    earlystop = EarlyStopping(
        monitor=monitor, 
        patience=2,
        min_delta=0.0001
    )

    unfreeze_callback = UnfreezeLayerCallback(
        layer_ranges=[slice(287, 509), slice(154, 287), slice(66, 154) , slice(40, 66), slice(0, 40)],
        base_model_name=base_model.name, monitor='val_accuracy', mode='max', patience=2
    )

    model = get_model(base_model, target_size, dense_units, n_classes, dropout=0.2)
    model.summary()

    if weight_path is not None:
        model.load_weights(weight_path)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=init_lr),
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall(), 
            tf.keras.metrics.Precision()
        ]
    )

    print(f'\nStarted Training for {log_name}')
    hist = model.fit(
        x=train_generator,
        epochs=n_epochs,
        validation_data=val_generator,
        class_weight=class_weight,
        # steps_per_epoch=3,
        # validation_steps=3,
        # callbacks = [tensorboard, reduce_lr, checkpoint, earlystop],
        # callbacks = [reduce_lr, checkpoint],
        callbacks = [reduce_lr, checkpoint, unfreeze_callback],
        # callbacks = [reduce_lr, unfreeze_callback],
    )


# log_name = f'derm_clf__{base_model.name}__opt_adam__lr_{str(init_lr)}__batch_{str(batch_size)}__acc_optimized__dense_' + '_'.join([str(x) for x in dense_units]) + '__cls_weighted'
# log_name = f'derm_clf__{base_model.name}__opt_adam__lr_{str(init_lr)}__batch_{str(batch_size)}__acc_optimized__dense_' + '_'.join([str(x) for x in dense_units]) + '__cls_weighted__clr_augmented'
log_name = f'derm_clf__{base_model.name}__opt_adam__lr_{str(init_lr)}__batch_{str(batch_size)}__acc_optimized__dense_' + '_'.join([str(x) for x in dense_units]) + '__clr_augmented__layer_freezed_2'
# log_name = f'derm_clf__{base_model.name}__opt_adam__lr_{str(init_lr)}__batch_{str(batch_size)}__acc_optimized__dense_' + '_'.join([str(x) for x in dense_units]) + '__w_img_name'

BASE_MODEL = EfficientNetV2S(include_top=False, input_shape=(384, 384, 3), weights="imagenet", pooling='avg')

training_config(
    base_model=BASE_MODEL, 

    dense_units=[64], 
    target_size=(384, 384, 3), 
    n_classes=3, 
    n_epochs=50, 
    init_lr=0.00025,
    batch_size=BATCH_SIZE, 
    class_weight=None, 
    weight_path=None, 
    monitor='val_accuracy'
)