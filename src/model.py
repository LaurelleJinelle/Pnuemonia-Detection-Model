# src/model.py

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from src.preprocessing import get_train_val_datasets, IMG_SIZE, BATCH_SIZE
from src.prediction import load_model  # optional if you want warm-loading

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_model(num_classes: int):
    """Builds a MobileNetV2 classification model."""

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def fine_tune(model, base_model):
    """Unfreeze top MobileNetV2 layers for fine-tuning."""
    base_model.trainable = True

    # Unfreeze top 40% of layers
    fine_tune_at = int(len(base_model.layers) * 0.6)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def retrain_model(retrain_data_dir: str):
    """
    Retrains model using new uploaded data + reload of original training dataset.
    Returns (new_model_path, history_dict).
    """

    # -----------------------------
    # 1. Load dataset for retraining
    # -----------------------------
    print("[INFO] Loading retraining dataset:", retrain_data_dir)

    train_ds, val_ds = get_train_val_datasets(retrain_data_dir)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # -----------------------------
    # 2. Build a fresh base model
    # -----------------------------
    print("[INFO] Building model...")
    model = build_model(num_classes)
    base_model = model.layers[2]  # MobileNetV2 block

    # -----------------------------
    # 3. TRAIN PHASE 1 — frozen base
    # -----------------------------
    print("[INFO] Training initial classifier layers...")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ]
    )

    # -----------------------------
    # 4. TRAIN PHASE 2 — fine-tune deeper layers
    # -----------------------------
    print("[INFO] Fine-tuning...")

    model = fine_tune(model, base_model)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            )
        ]
    )

    # -----------------------------
    # 5. Save new SavedModel
    # -----------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)

    new_model_path = os.path.join("models", "models/mobilenet_pneumonia_fixed.h5")
    model.save(new_model_path)

    # -----------------------------
    # 6. Return history to API
    # -----------------------------
    full_history = {
        "phase1": history1.history,
        "phase2": history2.history,
        "classes": class_names,
    }


    return new_model_path, full_history

