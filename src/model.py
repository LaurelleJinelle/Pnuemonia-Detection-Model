import os
import tensorflow as tf
from tensorflow.keras import layers, models

from src.preprocessing import get_train_val_datasets, IMG_SIZE, BATCH_SIZE

# Absolute models dir, e.g. /app/models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_model(num_classes: int):
    """
    Build a MobileNetV2 classification model.
    Returns (model, base_model) so we can fine-tune later.
    """
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
    """
    Unfreeze top MobileNetV2 layers for fine-tuning.
    """
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
    """
    Build callbacks list depending on whether we have validation data.
    If no validation set, monitor training loss instead of val_loss.
    """
    if has_val:
        monitor = "val_loss"
    else:
        monitor = "loss"

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=2,
        restore_best_weights=True,
    )

    return [early_stop]


def retrain_model(retrain_data_dir: str):
    """
    Retrain model using the data in `retrain_data_dir`.

    Returns:
        new_model_path (str): path to the saved model (absolute)
        history_dict (dict): training history that can be JSON-encoded
    """
    # -----------------------------
    # 1. Load dataset for retraining
    # -----------------------------
    print("[INFO] Loading retraining dataset:", retrain_data_dir)

    train_ds, val_ds = get_train_val_datasets(retrain_data_dir)
    class_names = getattr(train_ds, "class_names", None)
    if class_names is None:
        # Fallback in case TF version doesn't attach class_names
        class_names = []

    num_classes = len(class_names) if class_names else 0
    print(f"[INFO] Classes ({num_classes}): {class_names}")

    if num_classes == 0:
        raise ValueError(
            "No classes were found in the retraining dataset. "
            "Make sure /retrain_data has subfolders per class."
        )

    has_val = val_ds is not None

    # -----------------------------
    # 2. Build a fresh base model
    # -----------------------------
    print("[INFO] Building model...")
    model, base_model = build_model(num_classes)

    # -----------------------------
    # 3. TRAIN PHASE 1 — frozen base
    # -----------------------------
    print("[INFO] Training initial classifier layers...")
    callbacks_phase1 = _make_callbacks(has_val)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds if has_val else None,
        epochs=5,
        callbacks=callbacks_phase1,
    )

    # -----------------------------
    # 4. TRAIN PHASE 2 — fine-tune deeper layers
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
    # 5. Save new model
    # -----------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)

# Create versioned folder
version_name = "mobilenet_pneumonia_finetuned_model"
new_model_path = os.path.join(MODELS_DIR, version_name)

# Save SavedModel directory
model.save(new_model_path, save_format="tf")

# Save label map
label_map_path = os.path.join(new_model_path, "label_map.json")
with open(label_map_path, "w") as f:
    json.dump(class_names, f)


    # -----------------------------
    # 6. Prepare history for API/Streamlit
    # -----------------------------
    full_history = {
        "phase1": {k: [float(v_) for v_ in v] for k, v in history1.history.items()},
        "phase2": {k: [float(v_) for v_ in v] for k, v in history2.history.items()},
        "classes": class_names,
        "has_validation": has_val,
    }

    return new_model_path, full_history
