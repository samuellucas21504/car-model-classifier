import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from training_config import save_history_to_csv

import numpy as np

print("Dispositivos GPU:", tf.config.list_physical_devices('GPU'))

# Diretórios da base de dados (ajuste os caminhos conforme sua estrutura de arquivos)
train_dir = "dataset/train"
test_dir  = "dataset/test"

# Parâmetros
IMG_HEIGHT, IMG_WIDTH = 240, 240
EPOCHS = 40
BATCH_SIZE = 16

# Gerador de dados de treinamento com augmentação e separação de validação
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,         # rotações aleatórias até 20 graus
    width_shift_range=0.1,     # deslocamento horizontal até 10%
    height_shift_range=0.1,    # deslocamento vertical até 10%
    zoom_range=0.1,            # zoom aleatório até 10%
    horizontal_flip=True,      # flip horizontal aleatório
    validation_split=0.1       # separa 10% dos dados de treino para validação
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

# Gerador para dados de validação (subconjunto de validação)
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Gerador para dados de teste (sem augmentação, apenas preprocess)
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

# Número de classes
num_classes = train_generator.num_classes

# =====================================================================
# CÁLCULO DOS PESOS DE CLASSE (class_weight) A PARTIR DO train_generator
# =====================================================================
y_train = train_generator.classes  # array com o rótulo inteiro de cada amostra
class_counts = np.bincount(y_train, minlength=num_classes)  # contagem por classe

total_samples = float(len(y_train))

# Fórmula: w_c = N / (K * n_c)
class_weight = {}
for cls_idx in range(num_classes):
    n_c = class_counts[cls_idx]
    if n_c > 0:
        class_weight[cls_idx] = total_samples / (num_classes * n_c)
    else:
        # Em teoria não deveria ocorrer, mas só por segurança:
        class_weight[cls_idx] = 0.0

print("Contagem por classe:", class_counts)
print("Pesos por classe:", class_weight)

# Carregar base pré-treinada (EfficientNetB1) sem a top layer
base_model = EfficientNetB1(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False  # baseline: congela a base

# Montar o modelo completo adicionando as camadas de classificação
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compilar o modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Visão geral da arquitetura
model.summary()

# Callbacks para parar cedo e salvar o melhor modelo
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-3,
)

checkpoint = ModelCheckpoint(
    'models/EfficientNetB1/EfficientNetB1_baseline_best.keras',
    monitor='val_accuracy',
    save_best_only=True
)

# Treinamento inicial (somente camadas do topo treináveis)
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight,
    callbacks=[early_stop, checkpoint, reduce_lr]
)
save_history_to_csv(history, stage="baseline")

# Salvar o modelo baseline (após early stopping, já com os melhores pesos restaurados)
model.save('models/EfficientNetB1/EfficientNetB1_baseline.keras')

# (Opcional) Avaliação rápida no conjunto de teste
print("Avaliação baseline no conjunto de teste:")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Baseline - loss: {test_loss:.4f} - acc: {test_acc:.4f}")
