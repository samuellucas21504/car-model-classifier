import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from training_config import save_history_to_csv

import numpy as np

print("Dispositivos GPU:", tf.config.list_physical_devices('GPU'))

# Diretórios da base de dados (devem ser os mesmos)
train_dir = "dataset/train"
test_dir  = "dataset/test"

# Parâmetros
IMG_HEIGHT, IMG_WIDTH = 240, 240
EPOCHS = 60
BATCH_SIZE = 12

# Gerador de dados de treinamento com augmentação e separação de validação
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.1
)

# Gerador para dados de treinamento (subconjunto de treinamento)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Gerador para dados de validação
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Gerador para dados de teste
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

# Opcional: obter o mapeamento de classes
class_indices = train_generator.class_indices
print("Classes mapeadas:", class_indices)

num_classes = train_generator.num_classes

# ==========================================================
# CÁLCULO DOS PESOS DE CLASSE (MESMO ESQUEMA DO BASELINE)
# ==========================================================
y_train = train_generator.classes
class_counts = np.bincount(y_train, minlength=num_classes)
total_samples = float(len(y_train))

class_weight = {}
for cls_idx in range(num_classes):
    n_c = class_counts[cls_idx]
    if n_c > 0:
        class_weight[cls_idx] = total_samples / (num_classes * n_c)
    else:
        class_weight[cls_idx] = 0.0

print("Contagem por classe:", class_counts)
print("Pesos por classe:", class_weight)

# =========================================
# CARREGAR MODELO BASELINE E AVALIAR
# =========================================
baseline = keras.models.load_model("models/EfficientNetB1/EfficientNetB1_baseline.keras")

# Garantir que está compilado antes de avaliar
baseline.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Avaliação do modelo baseline no conjunto de teste:")
baseline_loss, baseline_acc = baseline.evaluate(test_generator)
print(f"Baseline - loss: {baseline_loss:.4f} - acc: {baseline_acc:.4f}")

# =========================================
# FINE-TUNING
# =========================================
# Vamos fine-tunar o mesmo modelo 'baseline'
model = baseline

# Pegar a EfficientNetB1 dentro do Sequential (primeira layer)
base_model = model.layers[0]

# Descongelar a base inteira, depois re-congelar as camadas iniciais
base_model.trainable = True

# Exemplo: só deixar as últimas 20 camadas da EfficientNet treináveis
for layer in base_model.layers[:-40]:
    layer.trainable = False

print("Número de camadas na base:", len(base_model.layers))
print("Camadas descongeladas (últimas 40):")
for layer in base_model.layers[-40:]:
    print("  ", layer.name)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop_ft = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=3e-6,
)

checkpoint_ft = ModelCheckpoint(
    "models/EfficientNetB1/EfficientNetB1_finetuned_best.keras",
    monitor="val_loss",
    save_best_only=True,
)

history_ft = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight,  # <<< mantém o balanceamento também no fine-tuning
    callbacks=[early_stop_ft, checkpoint_ft, reduce_lr],
)
save_history_to_csv(history_ft, stage="finetune")

# Salvar o último modelo fine-tunado (com os melhores pesos restaurados do early stopping)
model.save("models/EfficientNetB1/EfficientNetB1_finetuned_last.keras")
