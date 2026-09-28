# NumPy e Dados Numéricos

## Objetivo: 
Finalizar todo tema e criar um arquivo .ipynb no Google Colab e deixá-lo disponível no Github

## 1 Introdução ao NumPy e Arrays 
###  O que é NumPy e por que usá-lo 


### 1.1 O que é o NumPy?
O **NumPy** (abreviação de *Numerical Python*) é a biblioteca fundamental do ecossistema Python para computação científica, análise de dados e Inteligência Artificial. Criado em 2005 por Travis Oliphant, o NumPy fornece o objeto **`ndarray`** (*N-dimensional array*), uma estrutura de dados de alto desempenho projetada para armazenar e manipular grandes volumes de dados numéricos em arranjos multidimensionais (vetores, matrizes e tensores).

Embora seja utilizado diretamente em Python, a maior parte do código interno responsável pelo processamento pesado é escrita em linguagens de baixo nível compiladas, como **C e C++** . Isso permite executar operações matemáticas complexas com velocidade incomparável em relação ao código Python puro.

---

### 1.1.2 Por Que Usar o NumPy?

### 1.1.2.1 Desempenho e Velocidade Extrema
- **Execução em C:** As operações do NumPy são executadas por rotinas compiladas em C e C++, eliminando a sobrecarga de interpretação do Python.
- **Ganhos de Velocidade:** Em cálculos numéricos e manipulação de matrizes, o NumPy pode ser de **50 a mais de 100 vezes mais rápido** que as listas nativas do Python.
- **Paralelismo SIMD:** O NumPy aproveita instruções SIMD (*Single Instruction, Multiple Data*) das arquiteturas modernas de CPU para processar múltiplos elementos simultaneamente em hardware.

### 1.1.2.2  Armazenamento Homogêneo e Eficiência de Memória
- **Armazenamento Contíguo:** Enquanto as listas do Python armazenam ponteiros para objetos espalhados pela memória, o NumPy armazena elementos em **blocos contíguos de memória** .
- **Localidade de Referência:** Esse arranjo contíguo otimiza a *localidade de referência*, permitindo que a CPU carregue e processe blocos inteiros de dados na memória cache de forma ultraeficiente.
- **Dados Homogêneos:** Todos os elementos de um `ndarray` devem ser rigorosamente do mesmo tipo numérico (como `int32`, `float64`), o que economiza memória ao eliminar metadados individuais por elemento. Caso tipos mistos sejam fornecidos, o NumPy realiza *upcasting* automático para manter a consistência.

### 1.1.2.3 Operações Vetorizadas (Vetorização)
- **Eliminação de Loops Manuais:** Com a vetorização, operações aritméticas e matemáticas são aplicadas diretamente a um array inteiro de uma só vez.
- **Código Limpo e Conciso:** Substitui estruturas complexas e lentas de repetição (`for` loops) por sintaxe matemática direta e legível.

### 1.1.2.4 Funções Universais (*ufuncs*) e *Broadcasting*
- **ufuncs:** Funções altamente otimizadas implementadas em C que realizam operações elemento a elemento sobre arrays.
- **Broadcasting:** Mecanismo avançado que permite realizar operações aritméticas entre arrays de dimensões ou formatos diferentes sem a necessidade de duplicar dados ou criar loops manuais.

### 1.1.2.5 Base do Ecossistema de Data Science e Inteligência Artificial
O NumPy serve como o motor infraestrutural subjacente para praticamente todas as bibliotecas de análise de dados e aprendizado de máquina em Python:
- **Pandas:** Utiliza arrays do NumPy para gerenciar colunas em DataFrames.
- **SciPy:** Constrói algoritmos avançados de física, engenharia e estatística sobre estruturas NumPy [15, 19].
- **Scikit-Learn, TensorFlow e PyTorch:** Utilizam o NumPy e conceitos de tensores para pré-processamento de dados e treinamento de modelos de IA.

---

## 1.1.3 Comparativo: Listas do Python vs. Arrays do NumPy

| Recurso | Listas Nativas do Python | Arrays do NumPy (`ndarray`) |
| :--- | :--- | :--- |
| **Tipo de Dado** | Heterogêneo (aceita diferentes tipos no mesmo objeto) | Homogêneo (todos os elementos possuem o mesmo tipo) |
| **Layout de Memória** | Elementos dispersos conectados por ponteiros | Bloco contíguo de memória otimizado |
| **Desempenho** | Mais lento em cálculos por dependência de loops e tipagem dinâmica  | Altíssimo desempenho (código compilado em C e SIMD)  |
| **Operações Aritméticas** | Exigem loops explícitos ou list comprehensions  | Vetorizadas diretamente elemento a elemento (`A + B`)  |
| **Uso Principal** | Armazenamento flexível de propósito geral  | Computação científica, matrizes, Data Science e IA |

---


# 1.2 Criação de Arrays no NumPy

O NumPy oferece diversas funções nativas para criar objetos `ndarray` de forma eficiente, divididas em criação a partir de dados existentes, geradores de sequências, matrizes de inicialização e geradores de dados aleatórios.

---

## 1.2.1 Criação a partir de Estruturas Nativas (`np.array`)

A função `np.array()` converte sequências do Python (como listas e tuplas) em objetos `ndarray`.

```python
import numpy as np

# Vetor 1D a partir de uma lista
vetor = np.array([10, 20, 30, 40])

# Matriz 2D a partir de listas aninhadas
matriz = np.array([[1, 2, 3], [4, 5, 6]])

# Definindo o tipo de dado (dtype) explicitamente
array_float = np.array([1, 2, 3], dtype='float32')
array_datas = np.array(['2026-01-01', '2026-02-01'], dtype='datetime64')

print("Vetor:", vetor)
print("Matriz:\n", matriz)
print("Tipos:", array_float.dtype, "|", array_datas.dtype)
```

---

## 1.2.2 Sequências Numéricas (`np.arange` e `np.linspace`)

Quando precisamos de sequências ordenadas ou intervalos numéricos contínuos, utilizam-se `np.arange` e `np.linspace`.

* **`np.arange(start, stop, step)`**: Funciona como o `range()` do Python, mas retorna um array. O valor final (`stop`) é exclusivo.
* **`np.linspace(start, stop, num)`**: Gera uma quantidade fixa (`num`) de valores igualmente espaçados. O valor final (`stop`) é inclusivo por padrão.

```python
# Sequência de 0 a 18 com passo de 2
sequencia = np.arange(0, 20, 2)

# 5 valores igualmente espaçados entre 0 e 1
intervalo = np.linspace(0, 1, 5)

print("np.arange:", sequencia)
print("np.linspace:", intervalo)
```

---

## 1.2.3 Matrizes de Inicialização e Preenchimento

Para preparar estruturas antes de realizar cálculos ou inicializar parâmetros de modelos:

* **`np.zeros(shape)`**: Cria um array preenchido com zeros.
* **`np.ones(shape)`**: Cria um array preenchido com uns.
* **`np.full(shape, fill_value)`**: Cria um array preenchido com um valor constante definido.
* **`np.empty(shape)`**: Aloca espaço em memória sem inicializar os valores (conteúdo indeterminado, porém mais rápido).

```python
# Matriz 2x3 de Zeros
zeros = np.zeros((2, 3))

# Matriz 3x2 de Uns
uns = np.ones((3, 2))

# Matriz 2x2 preenchida com o valor 7
constante = np.full((2, 2), 7)

print("Zeros:\n", zeros)
print("Uns:\n", uns)
print("Constante:\n", constante)
```

---

## 1.2.4. Matrizes Quadradas e Diagonais (`np.eye` e `np.identity`)

Utilizadas em álgebra linear e operações com matrizes:

* **`np.eye(N, M)`**: Cria uma matriz com 1s na diagonal principal e 0s nos demais elementos (permite formatos retangulares $N \times M$).
* **`np.identity(n)`**: Cria estritamente uma matriz identidade quadrada de dimensão $n \times n$.

```python
# Matriz Identidade 3x3
identidade = np.identity(3)

# Matriz 3x4 com diagonal de 1s
eye_retangular = np.eye(3, 4)

print("Identidade:\n", identidade)
print("Eye (3x4):\n", eye_retangular)
```

---

## 1.2.5. Geração de Dados Aleatórios (`np.random`)

O módulo `np.random` é essencial para simulações, amostragem e inicialização em Machine Learning.

* **`np.random.randint(low, high, size)`**: Gera números inteiros aleatórios dentro de um intervalo.
* **`np.random.rand(d0, d1, ...)`**: Gera valores decimais aleatórios com distribuição uniforme entre 0 e 1.
* **`np.random.choice(a, size)`**: Seleciona elementos aleatórios a partir de um array existente.

```python
# Matriz 3x3 de inteiros aleatórios entre 0 e 9
inteiros_rand = np.random.randint(0, 10, size=(3, 3))

# Vetor de 4 floats aleatórios entre 0 e 1
floats_rand = np.random.rand(4)

# Escolha aleatória de elementos
amostra = np.random.choice([10, 20, 30, 40, 50], size=3)

print("Inteiros Aleatórios:\n", inteiros_rand)
print("Floats Aleatórios:", floats_rand)
print("Amostra Aleatória:", amostra)
```


# 1.3 Entendendo o `dtype` no NumPy

O **`dtype`** (*data type*) é o atributo que define o tipo exato de dado e a quantidade de memória ocupada por cada elemento dentro de um `ndarray`. 

Como o NumPy exige que os arrays sejam **homogêneos** (todos os elementos compartilham o mesmo tipo de dado), o `dtype` garante alocação contígua de memória e alta eficiência de processamento. É possível definir o tipo durante a criação do array ou converter um array existente usando o método **`.astype()`**.

---

## 1.3.1 Exemplos Práticos de `dtype` e `.astype()`

### 1.3.1.1. Conversão para Valores Booleanos (`bool`)
Ao converter um array numérico para booleano com `.astype(bool)`, o valor `0` torna-se `False`, enquanto qualquer valor diferente de zero torna-se `True`. Essa técnica é amplamente utilizada para criar máscaras lógicas de filtragem.

```python
import numpy as np

# Array com valores inteiros (incluindo zeros)
dados_num = np.array([0, 1, 5, 0, 10])

# Conversão para tipo booleano
mascara_bool = dados_num.astype(bool)

print("Array original:", dados_num)
print("Array Booleano:", mascara_bool)
print("Tipo de dado:", mascara_bool.dtype)  # Output: bool
```

---

### 1.3.1.2. Trabalhando com Datas e Tempo (`datetime64`)
O NumPy possui suporte nativo a dados temporais com o tipo **`datetime64`**, permitindo especificar a precisão desejada, como dias (`[D]`), meses (`[M]`) ou anos (`[Y]`), além de realizar aritmética de datas com `np.timedelta64`.

```python
import numpy as np

# Criação de array de datas com precisão diária [D]
datas = np.array(['2026-01-01', '2026-02-01', '2026-03-01'], dtype='datetime64[D]')

# Operação aritmética com datas (adicionando 10 dias)
proximas_datas = datas + np.timedelta64(10, 'D')

print("Datas iniciais:", datas)
print("Datas + 10 dias:", proximas_datas)
print("Tipo de dado:", datas.dtype)  # Output: datetime64[D]
```

---


# 2 Dimensões, Shape, Indexação e Slicing no NumPy

Compreender a estrutura de um `ndarray` e saber navegar pelos seus elementos são habilidades fundamentais no NumPy. Este guia aborda a inspeção de dimensões, formato e tamanho de arrays, além das técnicas de acesso, fatiamento e modificação de dados em 1D e 2D.

---

## 2.1. Atributos Estruturais: `ndim`, `shape` e `size`

Antes de manipular um array, é essencial inspecionar sua estrutura através de seus atributos nativos.

### 2.1.1. `ndim` (Número de Dimensões / Eixos)
O atributo `.ndim` indica a quantidade de dimensões (ou eixos) do array.
* **1D (Vetor):** 1 eixo.
* **2D (Matriz):** 2 eixos (linhas e colunas).
* **3D+ (Tensor):** 3 ou mais eixos.

```python
import numpy as np

# Exemplo 1: Comparando dimensões de diferentes estruturas
vetor = np.array([10, 20, 30])
matriz = np.array([[1, 2], [3, 4]])
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print("Vetor ndim:", vetor.ndim)   # Output: 1
print("Matriz ndim:", matriz.ndim) # Output: 2
print("Tensor ndim:", tensor.ndim) # Output: 3
```

```python
# Exemplo 2: Verificando dimensões após alteração de formato
arr = np.arange(12) # Array 1D com 12 elementos
matriz_respeitada = arr.reshape(3, 4) # Transformado em Matriz 2D (3x4)

print("Array original ndim:", arr.ndim)                # Output: 1
print("Array reformatado ndim:", matriz_respeitada.ndim) # Output: 2
```

---

### 2.1.2. `shape` (Formato das Dimensões)
O atributo `.shape` retorna uma tupla de inteiros indicando o tamanho do array ao longo de cada eixo. Para uma matriz 2D, o formato é `(linhas, colunas)`.

```python
import numpy as np

# Exemplo 1: Identificando o formato de vetores e matrizes
vetor = np.array([1, 2, 3, 4, 5])
matriz = np.array([[10, 20, 30], [40, 50, 60]])

print("Shape do vetor:", vetor.shape)   # Output: (5,)  -> 1D com 5 elementos
print("Shape da matriz:", matriz.shape) # Output: (2, 3) -> 2 linhas e 3 colunas
```

```python
# Exemplo 2: Trabalhando com matrizes tridimensionais (Tensores)
# Formato: (blocos/fatias, linhas, colunas)
tensor_3d = np.zeros((2, 3, 4))

print("Shape do Tensor 3D:", tensor_3d.shape) # Output: (2, 3, 4)
```

---

### 2.1.3. `size` (Quantidade Total de Elementos)
O atributo `.size` retorna o número total de elementos armazenados no array (equivalente à multiplicação dos valores da tupla `.shape`).

```python
import numpy as np

# Exemplo 1: Calculando o total de elementos em matrizes
matriz_a = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3 = 6 elementos
matriz_b = np.ones((4, 5))                  # 4x5 = 20 elementos

print("Total elementos Matriz A:", matriz_a.size) # Output: 6
print("Total elementos Matriz B:", matriz_b.size) # Output: 20
```

```python
# Exemplo 2: Relação entre shape e size em reorganização de dados
dados = np.arange(24)
matriz_4x6 = dados.reshape(4, 6)

# O tamanho total de elementos permanece constante independente do formato
print("Tamanho do vetor original:", dados.size)         # Output: 24
print("Tamanho após reshape (4x6):", matriz_4x6.size)   # Output: 24
```

---

## 2.2. Indexação (Acesso a Elementos)

A indexação no NumPy utiliza base zero (`0`). Em arrays multidimensionais, o acesso é feito passando os índices separados por vírgula em um único par de colchetes: `array[linha, coluna]`.

### 2.2.1. Indexação 1D
```python
import numpy as np

arr_1d = np.array([10, 20, 30, 40, 50])

# Exemplo 1: Acesso simples por índices positivos e negativos
primeiro = arr_1d[0]   # Primeiro elemento (10)
ultimo = arr_1d[-1]    # Último elemento (50)
penultimo = arr_1d[-2] # Penúltimo elemento (40)

print("Primeiro:", primeiro, "| Último:", ultimo, "| Penúltimo:", penultimo)
```

```python
# Exemplo 2: Acesso múltiplo usando Fancy Indexing (lista de índices)
indices = [0, 2, 4]
subconjunto = arr_1d[indices]

print("Elementos nos índices [0, 2, 4]:", subconjunto) # Output: [10 30 50]
```

### 2.2.2. Indexação 2D
```python
import numpy as np

matriz_2d = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

# Exemplo 1: Acesso direto formato [linha, coluna]
elem_centro = matriz_2d[1, 1] # Segunda linha (índice 1), segunda coluna (índice 1) -> 50
elem_topo_dir = matriz_2d[0, 2] # Primeira linha (índice 0), terceira coluna (índice 2) -> 30

print("Elemento central (1,1):", elem_centro)
print("Topo direito (0,2):", elem_topo_dir)
```

```python
# Exemplo 2: Combinando índices positivos e negativos em matrizes
# Pegando o último elemento da última linha
ultimo_elemento = matriz_2d[-1, -1] # Output: 90

# Pegando o primeiro elemento da última linha
primeiro_ultima_linha = matriz_2d[-1, 0] # Output: 70

print("Último elemento (-1, -1):", ultimo_elemento)
print("Primeiro elemento da última linha (-1, 0):", primeiro_ultima_linha)
```

---

## 2.3. Slicing (Fatiamento)

O fatiamento extrai subconjuntos de um array usando a sintaxe `[início:fim:passo]`.
* `início`: Índice inicial (inclusivo).
* `fim`: Índice final (exclusivo).
* `passo`: Incremento entre os elementos (opcional, padrão é 1).

> ⚠️ **Importante:** O fatiamento no NumPy cria uma **Visão (View)** e não uma cópia do array original. Alterar a fatia altera o array de origem.

### 2.3.1. Slicing 1D
```python
import numpy as np

vetor = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

# Exemplo 1: Intervalo simples e uso de passo
sub_intervalo = vetor[2:6]     # Do índice 2 ao 5 -> [20, 30, 40, 50]
elementos_pares = vetor[::2]   # Do início ao fim, de 2 em 2 -> [0, 20, 40, 60, 80]

print("Fatia [2:6]:", sub_intervalo)
print("Passo de 2 [::2]:", elementos_pares)
```

```python
# Exemplo 2: Omissão de limites e inversão de vetor
primeiros_quatro = vetor[:4]  # Do início até o índice 3 -> [0, 10, 20, 30]
vetor_invertido = vetor[::-1] # Inverte todo o vetor

print("Primeiros 4 elementos [:4]:", primeiros_quatro)
print("Vetor invertido [::-1]:", vetor_invertido)
```

### 2.3.2. Slicing 2D
Em matrizes 2D, o fatiamento é aplicado separadamente nas linhas e colunas: `matriz[fatia_linhas, fatia_colunas]`.

```python
import numpy as np

matriz = np.array([
    [ 1,  2,  3,  4],
    [ 5,  6,  7,  8],
    [ 9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Exemplo 1: Extraindo submatrizes e colunas completas
submatriz_centro = matriz[1:3, 1:3] # Linhas 1 e 2, Colunas 1 e 2
terceira_coluna = matriz[:, 2]      # Todas as linhas (:), coluna no índice 2

print("Submatriz central 2x2:\n", submatriz_centro)
print("Terceira coluna como vetor 1D:", terceira_coluna)
```

```python
# Exemplo 2: Seleção alternada de linhas e colunas (com passo)
linhas_pares_colunas_impares = matriz[::2, 1::2]

print("Linhas pares (0, 2) e colunas ímpares (1, 3):\n", linhas_pares_colunas_impares)
```

---

## 2.4. Alteração de Elementos

Arrays no NumPy são **mutáveis**. Podemos alterar valores individuais via indexação ou modificar regiões inteiras via slicing.

### 2.4.1. Alteração via Indexação
```python
import numpy as np

# Exemplo 1: Alterando valores individuais em vetor e matriz
vetor = np.array([1, 2, 3, 4])
vetor[0] = 99 # Substitui o primeiro elemento

matriz = np.array([[10, 20], [30, 40]])
matriz[1, 0] = 300 # Substitui o elemento na linha 1, coluna 0

print("Vetor alterado:", vetor)
print("Matriz alterada:\n", matriz)
```

```python
# Exemplo 2: Atribuição pontual com múltiplos índices
matriz_dados = np.zeros((3, 3), dtype=int)

# Alterando elementos em posições específicas
matriz_dados[0, 0] = 1
matriz_dados[1, 1] = 5
matriz_dados[2, 2] = 9

print("Matriz diagonal alterada:\n", matriz_dados)
```

### 2.4.2. Alteração em Bloco (via Slicing) e o Conceito de View vs Copy
```python
import numpy as np

# Exemplo 1: Alteração em bloco de uma linha ou coluna inteira
matriz = np.array([
    [10, 10, 10],
    [20, 20, 20],
    [30, 30, 30]
])

# Zerando a primeira coluna inteira
matriz[:, 0] = 0

# Substituindo toda a última linha por [99, 99, 99]
matriz[-1, :] = 99

print("Matriz modificada em bloco:\n", matriz)
```

```python
# Exemplo 2: Demonstrando o efeito de View vs Cópia explícita (.copy())
original = np.array([1, 2, 3, 4, 5])

# Fatiamento cria uma VIEW (compartilha a mesma memória)
fatia_view = original[1:4]
fatia_view[0] = 999 # Altera também o array original!

print("Original após alterar a VIEW:", original) # Output: [1, 999, 3, 4, 5]

# Para evitar alterar o original, usa-se .copy()
original_preservado = np.array([1, 2, 3, 4, 5])
fatia_copia = original_preservado[1:4].copy()
fatia_copia[0] = 888

print("Original preservado com .copy():", original_preservado) # Output: [1, 2, 3, 4, 5]
```






___


## Referências

- **Bisht, K. S.** (2022). *NumPy: From Basic to Advance*.
- **Idris, I.** (2014). *Learning NumPy Array*. Packt Publishing.
- **Khonprakhon, S.** (2023). *NumPy Mastery: 150 Practical Examples in Python*.
- **NumPy Documentation**. *NumPy Official Technical Manuals and User Guides*.
- **Sohail, M.** (2025). *NumPy for Data Analysis and Data Science: A Complete Guide*.
- **Yildiz, M.** (2024). *Mastering NumPy: The Ultimate Guide to Data Manipulation in Python*.



## Contribuidores
| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/50632736?v=4" width=115><br><sub>Nunes</sub>](https://github.com/nunesinc) | 
| :---: | 
