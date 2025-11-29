import os
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def get_train_val_datasets(data_dir: str, val_split: float = 0.2):
    # Count images first
    total_images = sum(len(files) for _,_,files in os.walk(data_dir))

    # If dataset is too small, skip validation split
    if total_images < 8:   
        print("[INFO] Small dataset detected — skipping validation split.")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        return train_ds, None

    # Normal mode (with validation split)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    return train_ds, val_ds

def get_dataset_from_directory(data_dir: str, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )

def preprocess_for_prediction(img_bytes: bytes):
    try:
        img = tf.image.decode_jpeg(img_bytes, channels=3)
    except:
        img = tf.image.decode_png(img_bytes, channels=3)

    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    img = tf.expand_dims(img, 0)  

    return img


