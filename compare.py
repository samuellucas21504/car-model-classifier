import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

train_dir = "dataset/train"
test_dir  = "dataset/test"

IMG_HEIGHT, IMG_WIDTH = 240, 240
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

baseline = keras.models.load_model("models/EfficientNetB1/EfficientNetB1_baseline.keras")

baseline.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Avaliação do modelo baseline no conjunto de teste:")
baseline_loss, baseline_acc = baseline.evaluate(test_generator)
print(f"Baseline - loss: {baseline_loss:.4f} - acc: {baseline_acc:.4f}")

print("\nRecarregando melhor modelo fine-tunado para avaliação...")
ft_best = keras.models.load_model("models/EfficientNetB1/EfficientNetB1_finetuned_best.keras")

ft_best.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("Avaliação do melhor modelo fine-tuned no conjunto de teste:")
ft_loss, ft_acc = ft_best.evaluate(test_generator)
print(f"Fine-tuned - loss: {ft_loss:.4f} - acc: {ft_acc:.4f}")

print("\nResumo:")
print(f"Baseline  - acc teste: {baseline_acc:.4f}")
print(f"Fine-tune - acc teste: {ft_acc:.4f}")
