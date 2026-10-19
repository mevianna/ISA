# Operadores Convolucionais

## O que são operadores convolucionais?
Os operadores convolucionais são utilizados em redes neurais convolucionais (CNNs) para processar imagens e identificar características relevantes em seus pixels. Para isso, filtros percorrem pequenas regiões da imagem, realizando operações matemáticas que permitem destacar padrões como bordas, texturas e formas. O resultado desse processo depende da maneira como os filtros são aplicados e dos parâmetros utilizados na operação.

> **Nota:** Redes Neurais Convolucionais (CNNs) são um tipo de rede neural utilizado principalmente no processamento e na análise de imagens. Sua principal característica é o uso de operações convolucionais para identificar padrões visuais, como bordas, texturas e formas, que podem ser utilizados em tarefas como classificação de imagens e detecção de objetos.

## Kernels (filtros convolucionais)
Os kernels são pequenas matrizes numéricas utilizadas na operação de convolução para analisar regiões de uma imagem. Durante o processamento, os valores do kernel são multiplicados pelos valores dos pixels correspondentes, e os resultados dessas multiplicações são somados, produzindo um único valor para cada posição analisada. Esse processo é repetido ao longo da imagem, permitindo identificar diferentes características visuais.

Em uma CNN, os valores dos kernels geralmente são ajustados durante o treinamento da rede, de modo que os filtros aprendam a reconhecer padrões relevantes para a tarefa proposta.

## Stride
O stride determina o número de pixels que o kernel se desloca a cada operação de convolução. Quando o *stride* é igual a 1, o filtro avança um pixel por vez, analisando regiões próximas da imagem. Já com stride igual a 2, o deslocamento passa a ser de dois pixels, reduzindo a quantidade de regiões analisadas e, consequentemente, as dimensões do mapa de características resultante. Dessa forma, o valor de stride influencia tanto o nível de detalhamento da análise quanto o tamanho da saída da camada convolucional.

## Padding
O padding consiste na adição de valores ao redor das bordas da imagem antes da aplicação do kernel. Geralmente, utiliza-se o preenchimento com zeros (*zero-padding*) para controlar as dimensões do mapa de características resultante e permitir que as regiões próximas às bordas sejam consideradas durante a convolução. Sem esse preenchimento, a saída tende a apresentar dimensões menores que as da imagem original. A quantidade de padding aplicada influencia, portanto, o tamanho da saída e a maneira como as informações das bordas são processadas.

## Mapas de Características (*Feature Maps*)
Os mapas de características (*feature maps*) são matrizes resultantes da aplicação de um kernel sobre uma imagem ou sobre os mapas produzidos por uma camada anterior. Cada valor dessas matrizes corresponde ao resultado da operação de convolução em uma determinada região da entrada. Os valores obtidos indicam a resposta do filtro aos padrões presentes nessas regiões, permitindo representar características como bordas, texturas e formas. Como uma camada convolucional pode utilizar vários kernels, ela também pode produzir diferentes mapas de características, cada um associado às respostas de um filtro específico.

## Exemplo Prático de Convolução

Considere uma imagem representada por uma matriz 4 × 4 e um *kernel* 3 × 3. Para simplificar o exemplo, será utilizado *stride* igual a 1 e *padding* igual a 0, ou seja, sem preenchimento nas bordas da imagem.

**Matriz de entrada:**

| 1 | 2 | 0 | 1 |
|---|---|---|---|
| 0 | 1 | 2 | 1 |
| 1 | 0 | 1 | 2 |
| 2 | 1 | 0 | 1 |

**Kernel:**

| 1 | 0 | 1 |
|---|---|---|
| 0 | 1 | 0 |
| 1 | 0 | 1 |

Durante a convolução, o *kernel* percorre a matriz de entrada e, em cada posição, multiplica os seus valores pelos elementos correspondentes da região analisada (*receptive field*). Em seguida, os produtos são somados, gerando um único valor para essa posição do mapa de características.

---

### 1. Primeira posição (Superior Esquerda)

A região analisada da matriz de entrada é:

| 1 | 2 | 0 |
|---|---|---|
| 0 | 1 | 2 |
| 1 | 0 | 1 |

A aplicação do *kernel* resulta no seguinte cálculo:

(1 × 1) + (2 × 0) + (0 × 1)  
+ (0 × 0) + (1 × 1) + (2 × 0)  
+ (1 × 1) + (0 × 0) + (1 × 1) = 4

---

### 2. Segunda posição (Superior Direita)

Deslocando o *kernel* 1 pixel para a direita (*stride* = 1), a região analisada passa a ser:

| 2 | 0 | 1 |
|---|---|---|
| 1 | 2 | 1 |
| 0 | 1 | 2 |

A aplicação do *kernel* resulta no seguinte cálculo:

(2 × 1) + (0 × 0) + (1 × 1)
+ (1 × 0) + (2 × 1) + (1 × 0)
+ (0 × 1) + (1 × 0) + (2 × 1) = 7

---

### 3. Terceira posição (Inferior Esquerda)

Deslocando o *kernel* 1 pixel para baixo e voltando à primeira coluna, a região analisada passa a ser:

| 0 | 1 | 2 |
|---|---|---|
| 1 | 0 | 1 |
| 2 | 1 | 0 |

A aplicação do *kernel* resulta no seguinte cálculo:

(0 × 1) + (1 × 0) + (2 × 1)
+ (1 × 0) + (0 × 1) + (1 × 0)
+ (2 × 1) + (1 × 0) + (0 × 1) = 4

---

### 4. Quarta posição (Inferior Direita)

Deslocando o *kernel* 1 pixel para a direita na linha inferior, a região analisada passa a ser:

| 1 | 2 | 1 |
|---|---|---|
| 0 | 1 | 2 |
| 1 | 0 | 1 |

A aplicação do *kernel* resulta no seguinte cálculo:

(1 × 1) + (2 × 0) + (1 × 1)
+ (0 × 0) + (1 × 1) + (2 × 0)
+ (1 × 1) + (0 × 0) + (1 × 1) = 5

---

### Resultado Final
Como a entrada possui dimensões 4 × 4, o *kernel* possui dimensões 3 × 3 e o *padding* é igual a 0, a dimensão de saída é dada por:

O = (4 - 3 + 2 × 0) / 1 + 1 = 2

O resultado é uma matriz 2 × 2.

**Mapa de características (*feature map*):**

| 4 | 7 |
|---|---|
| 4 | 5 |

Esse exemplo ilustra como a operação de convolução transforma a matriz de entrada em um mapa de características. Nas CNNs, diferentes *kernels* permitem identificar padrões distintos na imagem, gerando mapas que servem como entrada para as camadas subsequentes e contribuem para a extração de características relevantes ao longo da rede.

## Conclusão
Os operadores convolucionais permitem que as CNNs identifiquem características relevantes em imagens por meio da aplicação de filtros sobre regiões da entrada. Nesse processo, os *kernels* realizam as operações matemáticas, enquanto o *stride* e o *padding* determinam como o filtro percorre a imagem e como suas bordas são tratadas. Como resultado, são gerados os mapas de características (*feature maps*), que representam as respostas dos filtros aos padrões encontrados. Esses mapas são utilizados pelas camadas seguintes da rede, contribuindo para tarefas de visão computacional, como classificação de imagens e detecção de objetos.


## Referências
[1] O que é uma rede neural convolucional? Disponível em: <https://cloud.google.com/discover/what-are-convolutional-neural-networks?hl=pt-BR>. Acesso em: 19 out. 2026.

[2] O que são redes neurais convolucionais? Disponível em: <https://www.ibm.com/br-pt/think/topics/convolutional-neural-networks>. Acesso em: 19 out. 2026.

[3] Disponível em: <https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37/>. Acesso em: 19 out. 2026.




