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
