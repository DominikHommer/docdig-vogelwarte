"""Train the CRNN+CTC digit model (Digit2.ipynb, but as a runnable script).

Reads DIDA from `Digit-Recognition/DIDA/DIDA_1/` plus the label CSV
`Digit-Recognition/DIDA_12000_String_Digit_Labels.xls`, trains a small
Conv+BiLSTM+CTC model, and writes the inference subgraph to
`config/digit_model.keras` so the DigitRecognizer module can find it
automatically.

Usage:

    .venv/bin/python tools/train_digit_model.py            # full run (default 15 epochs)
    .venv/bin/python tools/train_digit_model.py --epochs 5  # quick smoke training

The CTC loss layer is wrapped in `tf.keras.layers.Lambda` so the model can be
serialised with `compile=False` and loaded back without custom_objects.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


ROOT = Path(__file__).resolve().parent.parent
DIDA_IMAGES = ROOT / "Digit-Recognition" / "DIDA" / "DIDA_1"
DIDA_LABELS = ROOT / "Digit-Recognition" / "DIDA_12000_String_Digit_Labels.xls"
OUT_PATH = ROOT / "config" / "digit_model.keras"

IMG_WIDTH = 160
IMG_HEIGHT = 40
CHARACTERS = list("0123456789")


def _load_labels() -> pd.DataFrame:
    df = pd.read_csv(DIDA_LABELS, header=None, encoding="utf-8-sig")
    df.columns = ["filename", "label"]
    df["filename"] = df["filename"].astype(str)
    df["label"] = df["label"].astype(str)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def _make_lookups():
    char_to_num = tf.keras.layers.StringLookup(
        vocabulary=CHARACTERS, mask_token=None, num_oov_indices=0
    )
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        mask_token=None,
        num_oov_indices=0,
        invert=True,
    )
    return char_to_num, num_to_char


def _make_dataset(df: pd.DataFrame, char_to_num, batch_size: int) -> tf.data.Dataset:
    img_paths = df["filename"].apply(
        lambda x: str(DIDA_IMAGES / f"{x}.jpg")
    ).values
    labels = df["label"].values
    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def _map(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = tf.cast(img, tf.float32) / 255.0

        label_chars = tf.strings.unicode_split(label, input_encoding="UTF-8")
        label_idx = char_to_num(label_chars)
        return img, label_idx

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([IMG_HEIGHT, IMG_WIDTH, 1], [None]),
        padding_values=(0.0, tf.constant(-1, dtype=tf.int64)),
        drop_remainder=True,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def _build_inference_model() -> tf.keras.Model:
    """Image-only Conv+BiLSTM model. Output: (B, T, num_classes+1) softmax."""
    inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    new_shape = (IMG_WIDTH // 4, (IMG_HEIGHT // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    out = layers.Dense(len(CHARACTERS) + 1, activation="softmax", name="logits")(x)
    return tf.keras.Model(inputs=inp, outputs=out, name="digit_inference")


def _ctc_loss(args):
    y_pred, labels = args
    label_len = tf.reduce_sum(
        tf.cast(tf.not_equal(labels, -1), dtype=tf.int32), axis=-1, keepdims=True
    )
    input_len = tf.ones(shape=(tf.shape(y_pred)[0], 1), dtype=tf.int32) * tf.shape(y_pred)[1]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_len, label_len)


def _build_training_model(inference_model: tf.keras.Model) -> tf.keras.Model:
    labels = layers.Input(name="label", shape=(None,), dtype="int64")
    loss = layers.Lambda(_ctc_loss, name="ctc_loss")([inference_model.output, labels])
    return tf.keras.Model(inputs=[inference_model.input, labels], outputs=loss)


def _format_batch(imgs, lbls):
    return ({"image": imgs, "label": lbls}, tf.zeros((tf.shape(imgs)[0], 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--quick", action="store_true",
        help="Use only 2000 samples and 3 epochs for a smoke run.",
    )
    args = parser.parse_args()

    if not DIDA_IMAGES.exists() or not DIDA_LABELS.exists():
        print(f"[train_digit_model] Missing DIDA data at {DIDA_IMAGES} / {DIDA_LABELS}")
        return 1

    print(f"[train_digit_model] Loading labels …")
    df = _load_labels()
    if args.quick:
        df = df.head(2000)
        args.epochs = min(args.epochs, 3)
        print("[train_digit_model] Quick mode: 2000 samples, 3 epochs")
    print(f"[train_digit_model] {len(df)} labelled samples available")

    char_to_num, num_to_char = _make_lookups()
    split = int(len(df) * (1 - args.val_split))
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    print(f"[train_digit_model] train={len(train_df)}  val={len(val_df)}")

    train_ds = _make_dataset(train_df, char_to_num, args.batch_size).map(_format_batch)
    val_ds = _make_dataset(val_df, char_to_num, args.batch_size).map(_format_batch)

    inference_model = _build_inference_model()
    training_model = _build_training_model(inference_model)
    training_model.compile(optimizer="adam", loss=lambda yt, yp: yp)

    inference_model.summary()

    history = training_model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs, verbose=1,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save the inference-only sub-model so we can load it back without
    # the CTC lambda — DigitRecognizer expects a single-input keras model.
    inference_model.save(OUT_PATH)
    print(f"[train_digit_model] Saved inference model -> {OUT_PATH}")

    # Quick sanity check: decode one batch on val and print the first 5 preds.
    print("\nSanity check on validation batch:")
    for batch in val_ds.take(1):
        imgs = batch[0]["image"]
        lbls = batch[0]["label"]
        preds = inference_model(imgs)
        input_len = np.ones(preds.shape[0]) * preds.shape[1]
        decoded = tf.keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
        for i in range(min(5, preds.shape[0])):
            res = decoded[i]
            res = tf.gather(res, tf.where(res != -1))[:, 0]
            text = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8", errors="ignore")
            truth_idx = lbls[i].numpy()
            truth = "".join(
                num_to_char(idx).numpy().decode("utf-8", errors="ignore")
                for idx in truth_idx if idx != -1
            )
            print(f"  truth={truth!r}  pred={text!r}")
        break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
