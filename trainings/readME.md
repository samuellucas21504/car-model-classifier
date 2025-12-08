## Versão 0.1
Tanto para baseline quanto para finetune usei esses parâmetros:

```py
preprocessing_function=preprocess_input,
rotation_range=20,         # rotações aleatórias até 20 graus
width_shift_range=0.1,     # deslocamento horizontal até 10%
height_shift_range=0.1,    # deslocamento vertical até 10%
zoom_range=0.1,            # zoom aleatório até 10%
horizontal_flip=True,      # flip horizontal aleatório
validation_split=0.1       # separa 10% dos dados de treino para validação
```

### Compare
`Baseline  - acc teste: 0.6140`\
`Fine-tune - acc teste: 0.6863`

## Versão 0.2

Tanto para baseline quanto para finetune usei esses parâmetros:
```py
preprocessing_function=preprocess_input,
rotation_range=10,         # rotações aleatórias até 20 graus
width_shift_range=0.2,     # deslocamento horizontal até 10%
height_shift_range=0.2,    # deslocamento vertical até 10%
zoom_range=0.05,            # zoom aleatório até 10%
horizontal_flip=True,      # flip horizontal aleatório
validation_split=0.1       # separa 10% dos dados de treino para validação
```

### Compare
`Baseline  - acc teste: 0.6140`\
`Fine-tune - acc teste: 0.6863`

## Versão 0.3

Foi feita uma alteração no finetune do modelo.

Foi feito um smooth label com as seguintes configurações:

```py
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=loss_fn,
    metrics=["accuracy"],
)
```

### Compare
`Baseline  - acc teste: 0.6140`
`Fine-tune - acc teste: 0.67`

## Versão 0.4

Após organização dos resultados do treinamento, ficou decidido que
a versão 0.1 teve melhor resultado na loss function e uma accuracy
parecida com a versão 0.3, portanto foi feito um rollback para essa versão.

Nessa versão, foi alterado o número de camadas que serão descongeladas no finetune,
de 20 foram aumentadas para 40 camadas.

### Compare

`Baseline  - acc teste: 0.6151`
`Fine-tune - acc teste: 0.6878`