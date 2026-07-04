"""Train the Sexe (m/w/none) classifier (port of M-W-Classification/cnn.ipynb).

Reads training images from a directory structure like::

    M-W-Classification/dataset/
        m/      maennchen1.png ...
        none/   leer1.png ...
        w/      weibchen1.png ...

Each subdirectory name becomes a class label (any set of subfolders works —
the labels are written to config/sexe_classes.json in softmax order). Run with:

    .venv/bin/python tools/train_sexe_model.py            # full training
    .venv/bin/python tools/train_sexe_model.py --quick    # 1 fold, 20 epochs

Output:
    - `config/sexe_model.keras`
    - `config/sexe_classes.json` (ordered list of class labels — pickle this
       so the SexeClassifier module knows the softmax order)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "M-W-Classification" / "dataset"
OUT_MODEL = ROOT / "config" / "sexe_model.keras"
OUT_CLASSES = ROOT / "config" / "sexe_classes.json"

IMG_HEIGHT = 150
IMG_WIDTH = 60


def _build_model(num_classes: int) -> tf.keras.Model:
    """CNN matching M-W-Classification/cnn.ipynb (tanh activations, L2 reg)."""
    l2 = tf.keras.regularizers.L2(l2=0.01)
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
            layers.Conv2D(32, (3, 3), activation="tanh"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="tanh"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="tanh"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation="tanh", kernel_regularizer=l2),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET,
        help=f"Directory of class subfolders. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--quick", action="store_true", help="20 epochs, no patience")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(
            f"[train_sexe_model] Dataset not found at {args.dataset}\n"
            f"Expected layout:\n"
            f"  {args.dataset}/\n"
            f"      m/     *.png  *.jpg\n"
            f"      none/  *.png  *.jpg\n"
            f"      w/     *.png  *.jpg\n"
            f"Drop your labelled cell crops in there and rerun."
        )
        return 1

    if args.quick:
        args.epochs = 20
        args.patience = 5

    train_dg = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=args.val_split
    )
    train_gen = train_dg.flow_from_directory(
        str(args.dataset),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        color_mode="grayscale",
    )
    val_gen = train_dg.flow_from_directory(
        str(args.dataset),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        color_mode="grayscale",
    )

    classes = sorted(train_gen.class_indices, key=train_gen.class_indices.get)
    print(f"[train_sexe_model] Classes: {classes}")
    print(f"[train_sexe_model] Train: {train_gen.samples}  Val: {val_gen.samples}")

    model = _build_model(num_classes=len(classes))
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[early_stop],
        verbose=1,
    )

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUT_MODEL)
    OUT_CLASSES.write_text(json.dumps(classes, ensure_ascii=False, indent=2))

    best_acc = float(max(history.history.get("val_accuracy", [0])))
    print(f"\n[train_sexe_model] Saved model -> {OUT_MODEL}")
    print(f"[train_sexe_model] Saved classes -> {OUT_CLASSES}")
    print(f"[train_sexe_model] Best validation accuracy: {best_acc:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
