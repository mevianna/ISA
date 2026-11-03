# ISA — Classificação de imagens em PyTorch

Este repositório reúne notebooks, em **PyTorch**, com pipelines de classificação de imagens: MNIST e
Fashion-MNIST, com foco no **monitoramento de métricas de treino e validação**, e **transfer learning**
com modelos pré-treinados no ImageNet no dataset **Animals-10 (Kaggle)**.

## Estrutura

| Pasta | Notebook | Dataset | Modelo salvo |
|---|---|---|---|
| `Mnist/` | `mnist_pytorch_vini.ipynb` | MNIST | `best_model.pth` |
| `Fashion_Mnist/` | `Fashion_Mnist_pytorch_vini.ipynb` | Fashion-MNIST | `best_model.pth` |
| `Transfer Learning/animals_10_kaggle - Resnet18/` | `animals_10_kaggle - Resnet18.ipynb` | Animals-10 | `best__resnet18.pt` |
| `Transfer Learning/animals_10_kaggle - MobileNetv2/` | `animals_10_kaggle - MobileNetv2.ipynb` | Animals-10 | `best__model.pt` |

## MNIST e Fashion-MNIST

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

## Resultados — MNIST e Fashion-MNIST

| Dataset | Parâmetros | Melhor val_accuracy | Acurácia no teste |
|---|---|---|---|
| MNIST | 4.366 | 0,9798 | **0,9797** |
| Fashion-MNIST | 59.546 | 0,8880 | **0,8876** |

## Transfer Learning — Animals-10 (Kaggle)

### Dataset

- **Fonte**: [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) (Kaggle), lido com
  `torchvision.datasets.ImageFolder`.
- **Classes (10)**: `butterfly`, `cat`, `chicken`, `cow`, `dog`, `elephant`, `horse`, `sheep`, `spider`,
  `squirrel`.
- **Tamanho**: 26.179 imagens, divididas aleatoriamente (seed 42) em **80% treino / 10% validação / 10%
  teste** → 20.943 / 2.617 / 2.619.
- **Desbalanceamento**: no treino, as classes variam de 1.133 (`elephant`) a 3.907 (`dog`) amostras. Os
  notebooks calculam `class_weights` inversamente proporcionais à frequência **apenas no split de treino**
  (estilo `balanced` do scikit-learn), evitando vazamento de validação/teste.
- **Pré-processamento (treino)**: `Resize(224×224)`, conversão para RGB, `RandomHorizontalFlip(0.1)`,
  `RandomVerticalFlip(0.1)`, `RandomAffine(±10°, translate=0.1, scale=0.9–1.1)`, `ToTensor` e normalização
  ImageNet (`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`).
- **Validação/teste**: apenas `Resize(224×224)`, RGB, `ToTensor` e a mesma normalização.
- **Caminho esperado**: `/home/jovyan/work/Datasets/animals_10_kaggle/animals_10_kaggle/raw-img` — ajuste a
  variável `PATH` conforme o seu ambiente.

### Transfer learning

Os dois notebooks partem de modelos **pré-treinados no ImageNet**, carregados via `timm`, e substituem a
camada de classificação original por uma nova cabeça com 10 saídas:

| Backbone (`timm`) | Cabeça substituída | Parâmetros totais | Checkpoint |
|---|---|---|---|
| `resnet18` | `fc` | 11.181.642 | `best__resnet18.pt` |
| `mobilenetv2_100` | `classifier` | 2.236.682 | `best__model.pt` |

- **Fine-tuning completo**: todo o backbone é treinado (sem congelamento), com `drop_rate=0.3` na cabeça.
- **LR discriminativo**: backbone com `1e-4` e cabeça nova com `1e-3`, via `AdamW` (`weight_decay=1e-4`).
- **Agendamento**: `OneCycleLR` (`max_lr=[1e-4, 1e-3]`, `div_factor=100`, `final_div_factor=100`),
  atualizado a cada batch.
- **Loss**: `CrossEntropyLoss` com `class_weights` e `label_smoothing=0.05`.
- **Estabilização**: gradient clipping em `1.0`, batch 64 e seed 42.
- **Seleção do modelo**: até 40 épocas, com early stopping (paciência 8) sobre a acurácia de validação; o
  melhor checkpoint é salvo e restaurado ao final.

### Resultados — Animals-10

| Modelo | Épocas executadas | Melhor val_acc | Acurácia no teste | Macro F1 |
|---|---|---|---|---|
| ResNet18 | 23 (early stopping) | 0,9744 | 0,9725 | 0,9705 |
| MobileNetV2 | 40 | **0,9782** | **0,9763** | **0,9744** |

O MobileNetV2 atinge acurácia ligeiramente superior com **~5× menos parâmetros** que o ResNet18.

### Avaliação e visualização

- Acurácia no test set e curvas de loss/acurácia (treino vs. validação).
- `classification_report` com precision, recall e F1 por classe.
- **Matriz de confusão** com `seaborn.heatmap`.
- Grade de 20 predições aleatórias do teste (título verde = acerto, vermelho = erro).

## Como executar

Abra cada notebook no Jupyter e execute as células em ordem. Requisitos:

- **MNIST/Fashion-MNIST**: `torch`, `torchvision`, `numpy`, `matplotlib`.
- **Transfer Learning**: `torch`, `torchvision`, `timm`, `numpy`, `matplotlib`, `scikit-learn`, `seaborn`,
  `tqdm`.

GPU CUDA é opcional (os notebooks detectam o dispositivo automaticamente), mas recomendada para o treino
dos modelos de transfer learning.
