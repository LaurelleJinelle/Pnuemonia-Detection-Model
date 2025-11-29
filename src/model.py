import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models

from src.preprocessing import get_train_val_datasets, IMG_SIZE, BATCH_SIZE

# Absolute models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_model(num_classes: int):
    """Build a MobileNetV2 classification model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
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
        metrics=["accuracy"],
    )
    return model, base_model


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
        metrics=["accuracy"],
    )
    return model


def _make_callbacks(has_val: bool):
    """Return callbacks that work safely with or without validation."""
    monitor = "val_loss" if has_val else "loss"

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=2,
        restore_best_weights=True,
    )
    return [early_stop]


def retrain_model(retrain_data_dir: str):
    """Retrain model and return (new_model_path, training_history)."""

    # -----------------------------
    # 1. Load dataset
    # -----------------------------
    print("[INFO] Loading retraining dataset:", retrain_data_dir)

    train_ds, val_ds = get_train_val_datasets(retrain_data_dir)

    class_names = getattr(train_ds, "class_names", [])
    num_classes = len(class_names)
    print(f"[INFO] Classes ({num_classes}): {class_names}")

    if num_classes == 0:
        raise ValueError(
            "Retraining aborted: No class folders found under /retrain_data."
        )

    has_val = val_ds is not None

    # -----------------------------
    # 2. Build model
    # -----------------------------
    print("[INFO] Building model...")
    model, base_model = build_model(num_classes)

    # -----------------------------
    # 3. Phase 1 – Train classifier head
    # -----------------------------
    print("[INFO] Training classifier head...")
    callbacks_phase1 = _make_callbacks(has_val)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds if has_val else None,
        epochs=5,
        callbacks=callbacks_phase1,
    )

    # -----------------------------
    # 4. Phase 2 – Fine-tuning
    # -----------------------------
    print("[INFO] Fine-tuning...")
    model = fine_tune(model, base_model)
    callbacks_phase2 = _make_callbacks(has_val)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds if has_val else None,
        epochs=5,
        callbacks=callbacks_phase2,
    )

    # -----------------------------
    # 5. Save model safely
    # -----------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)

    version_name = "mobilenet_pneumonia_finetuned_model"
    new_model_path = os.path.join(MODELS_DIR, version_name)

    print("[INFO] Saving model to:", new_model_path)
    model.save(new_model_path, save_format="tf")

    # Save class labels
    label_map_path = os.path.join(new_model_path, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(class_names, f)

    # -----------------------------
    # 6. Build JSON-serializable history
    # -----------------------------
    full_history = {
        "classes": class_names,
        "phase1": {k: [float(v_) for v_ in v] for k, v in history1.history.items()},
        "phase2": {k: [float(v_) for v_ in v] for k, v in history2.history.items()},
        "has_validation": has_val,
    }

    return new_model_path, full_history
