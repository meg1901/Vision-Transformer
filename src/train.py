from pathlib import Path
from tensorflow import keras

from .config import ViTConfig

def compile_model(model, config: ViTConfig):
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name="top_5_accuracy"
            ),
        ],
    )

def train_and_evaluate(
    model, x_train, y_train, x_test, y_test, config: ViTConfig,
    output_dir="outputs"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.weights.h5"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=config.validation_split,
        callbacks=callbacks,
        verbose=2,
    )

    if checkpoint.exists():
        model.load_weights(checkpoint)

    loss, top1, top5 = model.evaluate(x_test, y_test, verbose=0)
    return history, {
        "test_loss": float(loss),
        "test_accuracy": float(top1),
        "top_5_accuracy": float(top5),
    }
