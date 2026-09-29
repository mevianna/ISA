# ISA — MNIST e Fashion-MNIST em PyTorch

Este repositório contém dois notebooks que reproduzem, em **PyTorch**, os pipelines dos scripts TensorFlow/Keras
(`mnist-small-model-tflite-int8.py` e `fashion-mnist-small-model-tflite-int8.py`), focando no
**monitoramento de métricas de treino e validação** — sem a parte de quantização/TFLite.

## Estrutura

| Pasta | Notebook | Dataset | Modelo salvo |
|---|---|---|---|
| `Mnist/` | `mnist_pytorch_vini.ipynb` | MNIST | `best_model.pth` |
| `Fashion_Mnist/` | `Fashion_Mnist_pytorch_vini.ipynb` | Fashion-MNIST | `best_model.pth` |

## O que foi feito

- **Dados**: download via `torchvision`, split 90/10 do treino para validação (54.000 / 6.000) + 10.000 de teste.
  Aumento de dados com `RandomAffine` (rotação ±36°, translação e zoom de 10%), equivalente aos
  `RandomRotation/Zoom/Translation` do Keras.
- **Callbacks do Keras reimplementados em PyTorch**:
  - `History` — histórico de loss/accuracy (treino e validação) e LR por época;
  - `ModelCheckpoint` — salva o `best_model.pth` quando `val_accuracy` melhora;
  - `EarlyStopping` — paciência de 10 épocas, restaurando os melhores pesos;
  - `ReduceLROnPlateau` — reduz o LR por fator 0.1 após 5 épocas sem melhora (mín. `1e-6`).
- **Modelos CNN "balanced"** com BatchNorm, MaxPool, Global Average Pooling, Dropout e camadas densas.
- **Treino**: AdamW (`lr=0.01`, `weight_decay=1e-4`), `CrossEntropyLoss`, batch 64, até 100 épocas (seed 42).
- **Avaliação/visualização**: amostras do dataset, curvas de loss/accuracy, acurácia no teste e predições
  em amostras aleatórias (título verde = acerto, vermelho = erro).

## Resultados

| Dataset | Parâmetros | Melhor val_accuracy | Acurácia no teste |
|---|---|---|---|
| MNIST | 4.366 | 0,9798 | **0,9797** |
| Fashion-MNIST | 59.546 | 0,8880 | **0,8876** |

## Como executar

Abra cada notebook no Jupyter e execute as células em ordem. Requer `torch`, `torchvision`,
`numpy` e `matplotlib` (com GPU CUDA quando disponível).
