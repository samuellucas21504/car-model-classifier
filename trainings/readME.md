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
os pesos de 0.1 obtiveram melhor resultado, portanto houve um rollback dos pesos nessa etapa.

Nessa versão, foi alterado o número de camadas que serão descongeladas no finetune,
de 20 foram aumentadas para 40 camadas.

Notou-se uma pequena melhora no val accuracy do teste, portanto essa alteração foi mantida.

### Compare

`Baseline  - acc teste: 0.6151`
`Fine-tune - acc teste: 0.6878`


## Versão 0.5

Nessa versão foi alterado o learning rate do finetune de 1e-5 para 5e-5, além disso foi
feito o rollback do label_smoothing, a loss function usada agora é [categorical_crossentropy](https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/). 

Começar com um learning rate menor fez com que o modelo após o fine tune convergisse em um
val_acc menor, quase igual ao val_acc do baseline.

### Compare
`Baseline  - acc teste: 0.6151`
`Fine-tune - acc teste: 0.6199`

## Versão 0.6

Como demonstrado em 0.5, um learning rate menor fez com que o modelo tivesse um acc
parecido com o val_acc do baseline. Portanto, nessa versão subimos o lr para 1e-5
e adicionamos ReduceLROnPlateau com os seguintes parâmetros:

```py
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
```

Quando o modelo trava numa loss, ele diminui o LR sozinho, o que costuma melhorar fine-tuning em datasets pequenos.

Nota-se uma pequena melhora no val_acc do fine-tune 
ao aplicar essa mudança de LR.

### Compare
`Baseline  - acc teste: 0.6153`
`Fine-tune - acc teste: 0.6982`

## Versão 0.7

Essa versão alterou o modelo base de EfficientNetB0 para EfficientNetB1.

Nota-se uma melhora no valor final do fine-tune, sendo esse
o primeiro treinamento que superou a marca de 70% de acurácia no teste.

### Compare
`Baseline  - acc teste: 0.6160`
`Fine-tune - acc teste: 0.7181`

## Versäo 0.8

Nessa versão foi duplicado o número de épocas para 40 e reduzido o batch size pela metade, para 16.
Isso fez com que o tempo de processamento aumentasse em 4x, mas o modelo obteve uma melhora significativa no val_acc.

### Compare
`Baseline  - acc teste: 0.6328`
`Fine-tune - acc teste: 0.7687`

## Versão 0.9

Como na versão 0.8 o modelo obteve uma melhora significativa, nessa versão foi decidido aumentar o número de épocas para 60.
Foi diminuido o tamanho da batch de 16 para 12.

Além disso, ao ler a documentação do ReduceLROnPlateau, percebi que o min_lr não funcionava da maneira que imaginei,
portanto o reduzi de 1e-3 para 5e-4.

Nota-se que o 0.9, comparado ao 0.8, não obteve uma melhora significativa para justificar o aumento do tempo de treinamento,
portanto essa alteração não foi benéfica e será revertida na próxima versão.

### Compare
`Baseline  - acc teste: 0.6392`
`Fine-tune - acc teste: 0.747`

### Versão 1.0

Nessa versão foi revertido o número de épocas para 40 e o batch size para 16.
Além disso, foi alterado o modelo base para EfficientNetB4.

Notou-se um aumento significativo no tempo de treinamento e apenas uma pequena melhora
no val_acc do fine-tune.

Portanto é provável que a base de treinamento ou validação não esteja bem trabalhada
e para que vejamos melhorias significativas, será necessário trabalhar melhor os dados.

### Compare
`Baseline  - acc teste: 0.6805`
`Fine-tune - acc teste: 0.7793`