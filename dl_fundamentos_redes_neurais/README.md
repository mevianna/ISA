# Fundamentos de Redes Neurais: do Neurônio ao Gradiente

Este módulo apresenta os fundamentos necessários para entender como um modelo neural produz uma previsão, mede o próprio erro e ajusta seus parâmetros. O percurso começa no neurônio artificial e passa por funções de ativação, grafos computacionais, retropropagação e descida do gradiente.

O conteúdo foi produzido em português para a comunidade externa do RepoAI.

## Objetivos de aprendizagem

Ao concluir o módulo, você deverá ser capaz de:

- explicar a soma ponderada, o viés e as principais funções de ativação;
- entender por que uma rede sem ativações não lineares perde expressividade;
- reconhecer um DAG e interpretar um grafo computacional;
- acompanhar o *forward pass* e calcular gradientes com a regra da cadeia;
- diferenciar Gradient Descent em lote de Stochastic Gradient Descent;
- relacionar taxa de aprendizado, estabilidade e convergência.

## Roteiro de estudos

1. [Neurônio e funções de ativação](content/01_neuronio_e_funcoes_de_ativacao.md)
2. [DAGs e suas aplicações em Machine Learning](content/02_dags_em_machine_learning.md)
3. [Grafos computacionais e backpropagation](content/03_grafos_computacionais_e_backpropagation.md)
4. [Função de perda, gradiente e SGD](content/04_gradiente_e_sgd.md)
5. [Notebook: neurônio e funções de ativação](code/neuronio_e_funcoes_de_ativacao.ipynb)

O fluxo conceitual do módulo é:

```text
entradas → soma ponderada → ativação → previsão → perda
                                              ↓
atualização dos parâmetros ← gradientes ← backpropagation
```

## Estrutura

```text
dl_fundamentos_redes_neurais/
├── content/
│   ├── images/
│   │   └── colaboradores/
│   ├── 01_neuronio_e_funcoes_de_ativacao.md
│   ├── 02_dags_em_machine_learning.md
│   ├── 03_grafos_computacionais_e_backpropagation.md
│   └── 04_gradiente_e_sgd.md
├── code/
│   └── neuronio_e_funcoes_de_ativacao.ipynb
├── README.md
└── requirements.txt
```

## Pré-requisitos

- noções de álgebra básica;
- familiaridade com funções e derivadas;
- Python básico para executar o notebook;
- NumPy, Matplotlib e scikit-learn.

Para preparar o ambiente:

```bash
python -m pip install -r requirements.txt
```

## Autoria por conteúdo

| Conteúdo | Autoria original |
|---|---|
| Neurônio e funções de agregação | Beatriz Schuelter Tartare |
| Funções de ativação | Clara Marcela Grossl |
| Notebook de neurônio e ativações | Vinícius Muchulski |
| DAGs em Machine Learning | Adilson Vissoli e Arthur Bogoni |
| Grafos computacionais | Arthur Bogoni |
| Forward pass e backpropagation | Alice Motin |
| Função de perda, gradiente e convergência | Lucas Schemes |
| Funcionamento operacional do SGD | Daiana Brum |

## Colaboradores

| | | | |
|:---:|:---:|:---:|:---:|
| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/197432407?v=4" width="115" alt="Beatriz Schuelter Tartare"><br><sub>Beatriz Schuelter Tartare</sub>](https://github.com/beastartare) | [<img loading="lazy" src="https://avatars.githubusercontent.com/u/199311034?v=4" width="115" alt="Clara Marcela Grossl"><br><sub>Clara Marcela Grossl</sub>](https://github.com/Clara-M-Grossl) | [<img loading="lazy" src="https://avatars.githubusercontent.com/u/105316221?v=4" width="115" alt="Vinícius Muchulski"><br><sub>Vinícius Muchulski</sub>](https://github.com/vini-muchulski) | [<img loading="lazy" src="https://avatars.githubusercontent.com/u/207051125?v=4" width="115" alt="Arthur Bogoni"><br><sub>Arthur Bogoni</sub>](https://github.com/ArthurBogoni) |
| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/112569754?v=4" width="115" alt="Alice Motin"><br><sub>Alice Motin</sub>](https://github.com/AliceMotin) | <img loading="lazy" src="content/images/colaboradores/adilson_vissoli.svg" width="115" alt="Adilson Vissoli"><br><sub>Adilson Vissoli</sub> | <img loading="lazy" src="content/images/colaboradores/lucas_schemes.svg" width="115" alt="Lucas Schemes"><br><sub>Lucas Schemes</sub> | <img loading="lazy" src="content/images/colaboradores/daiana_brum.svg" width="115" alt="Daiana Brum"><br><sub>Daiana Brum</sub> |
