import argparse
import json
from dataclasses import replace
from pathlib import Path
import tensorflow as tf

from src.config import ViTConfig
from src.data import load_cifar10, build_augmentation
from src.model import create_vit_classifier
from src.plots import save_training_plots
from src.train import compile_model, train_and_evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config = ViTConfig()
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)

    tf.keras.utils.set_random_seed(config.seed)

    (x_train, y_train), (x_test, y_test) = load_cifar10()

    if args.smoke_test:
        x_train, y_train = x_train[:2048], y_train[:2048]
        x_test, y_test = x_test[:512], y_test[:512]
        config = replace(config, epochs=1, batch_size=64)

    augmentation = build_augmentation(x_train, config)
    model = create_vit_classifier(augmentation, config)
    compile_model(model, config)
    model.summary()

    history, results = train_and_evaluate(
        model, x_train, y_train, x_test, y_test, config
    )
    save_training_plots(history)

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print(f"Top-1 accuracy: {results['test_accuracy'] * 100:.2f}%")
    print(f"Top-5 accuracy: {results['top_5_accuracy'] * 100:.2f}%")

if __name__ == "__main__":
    main()
