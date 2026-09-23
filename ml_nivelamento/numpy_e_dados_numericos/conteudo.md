# NumPy e Dados Numéricos
## 1 Introdução ao NumPy e Arrays 
###  O que é NumPy e por que usá-lo 


### 1.1 O que é o NumPy?
O **NumPy** (abreviação de *Numerical Python*) é a biblioteca fundamental do ecossistema Python para computação científica, análise de dados e Inteligência Artificial. Criado em 2005 por Travis Oliphant, o NumPy fornece o objeto **`ndarray`** (*N-dimensional array*), uma estrutura de dados de alto desempenho projetada para armazenar e manipular grandes volumes de dados numéricos em arranjos multidimensionais (vetores, matrizes e tensores).

Embora seja utilizado diretamente em Python, a maior parte do código interno responsável pelo processamento pesado é escrita em linguagens de baixo nível compiladas, como **C e C++** . Isso permite executar operações matemáticas complexas com velocidade incomparável em relação ao código Python puro.

---

### 1.2 Por Que Usar o NumPy?

### 1.2.1 Desempenho e Velocidade Extrema
- **Execução em C:** As operações do NumPy são executadas por rotinas compiladas em C e C++, eliminando a sobrecarga de interpretação do Python.
- **Ganhos de Velocidade:** Em cálculos numéricos e manipulação de matrizes, o NumPy pode ser de **50 a mais de 100 vezes mais rápido** que as listas nativas do Python.
- **Paralelismo SIMD:** O NumPy aproveita instruções SIMD (*Single Instruction, Multiple Data*) das arquiteturas modernas de CPU para processar múltiplos elementos simultaneamente em hardware.

### 1.2.2  Armazenamento Homogêneo e Eficiência de Memória
- **Armazenamento Contíguo:** Enquanto as listas do Python armazenam ponteiros para objetos espalhados pela memória, o NumPy armazena elementos em **blocos contíguos de memória** .
- **Localidade de Referência:** Esse arranjo contíguo otimiza a *localidade de referência*, permitindo que a CPU carregue e processe blocos inteiros de dados na memória cache de forma ultraeficiente.
- **Dados Homogêneos:** Todos os elementos de um `ndarray` devem ser rigorosamente do mesmo tipo numérico (como `int32`, `float64`), o que economiza memória ao eliminar metadados individuais por elemento. Caso tipos mistos sejam fornecidos, o NumPy realiza *upcasting* automático para manter a consistência.

### 1.2.3 Operações Vetorizadas (Vetorização)
- **Eliminação de Loops Manuais:** Com a vetorização, operações aritméticas e matemáticas são aplicadas diretamente a um array inteiro de uma só vez.
- **Código Limpo e Conciso:** Substitui estruturas complexas e lentas de repetição (`for` loops) por sintaxe matemática direta e legível.

### 1.2.4 Funções Universais (*ufuncs*) e *Broadcasting*
- **ufuncs:** Funções altamente otimizadas implementadas em C que realizam operações elemento a elemento sobre arrays.
- **Broadcasting:** Mecanismo avançado que permite realizar operações aritméticas entre arrays de dimensões ou formatos diferentes sem a necessidade de duplicar dados ou criar loops manuais.

### 1.2.5 Base do Ecossistema de Data Science e Inteligência Artificial
O NumPy serve como o motor infraestrutural subjacente para praticamente todas as bibliotecas de análise de dados e aprendizado de máquina em Python:
- **Pandas:** Utiliza arrays do NumPy para gerenciar colunas em DataFrames.
- **SciPy:** Constrói algoritmos avançados de física, engenharia e estatística sobre estruturas NumPy [15, 19].
- **Scikit-Learn, TensorFlow e PyTorch:** Utilizam o NumPy e conceitos de tensores para pré-processamento de dados e treinamento de modelos de IA.

---

## 1.3 Comparativo: Listas do Python vs. Arrays do NumPy

| Recurso | Listas Nativas do Python | Arrays do NumPy (`ndarray`) |
| :--- | :--- | :--- |
| **Tipo de Dado** | Heterogêneo (aceita diferentes tipos no mesmo objeto) | Homogêneo (todos os elementos possuem o mesmo tipo) |
| **Layout de Memória** | Elementos dispersos conectados por ponteiros | Bloco contíguo de memória otimizado |
| **Desempenho** | Mais lento em cálculos por dependência de loops e tipagem dinâmica  | Altíssimo desempenho (código compilado em C e SIMD)  |
| **Operações Aritméticas** | Exigem loops explícitos ou list comprehensions  | Vetorizadas diretamente elemento a elemento (`A + B`)  |
| **Uso Principal** | Armazenamento flexível de propósito geral  | Computação científica, matrizes, Data Science e IA |

---

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
