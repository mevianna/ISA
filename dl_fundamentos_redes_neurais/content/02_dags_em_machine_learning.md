# DAG (Directed Acyclic Graph) e sua Aplicação em Machine Learning

## 1. Introdução

Os grafos são estruturas matemáticas amplamente utilizadas na computação para representar relações entre elementos. Eles aparecem em diversas áreas, como redes sociais, sistemas de recomendação, roteamento de internet, compiladores e inteligência artificial.

Dentro da teoria dos grafos, existe uma estrutura extremamente importante chamada DAG (Directed Acyclic Graph), ou Grafo Direcionado Acíclico. Esse tipo de grafo é fundamental para representar dependências e fluxos de execução em sistemas computacionais modernos.

Em Machine Learning e Deep Learning, DAGs são utilizados para modelar grafos computacionais, permitindo organizar operações matemáticas e otimizar o treinamento de modelos de inteligência artificial. Frameworks modernos, como TensorFlow e PyTorch, utilizam DAGs para representar o fluxo de dados e operações durante o treinamento de redes neurais.

Este trabalho apresenta os conceitos fundamentais dos DAGs, suas propriedades, aplicações computacionais e sua importância na área de Machine Learning, especialmente em problemas de classificação binária.

## 2. Conceito de DAG

DAG significa:

> Directed Acyclic Graph — Grafo Direcionado Acíclico

Um DAG é um tipo de grafo que possui duas características principais:

- As arestas possuem direção;
- Não existem ciclos.

Em um grafo direcionado, as conexões entre os nós possuem um sentido definido.

Exemplo:

```text
A → B → C
```

Nesse caso:

- A aponta para B;
- B aponta para C.

O grafo é considerado acíclico porque não existe um caminho que permita retornar ao nó inicial.

Exemplo de ciclo:

```text
A → B → C → A
```

Nesse caso existe um ciclo, portanto o grafo não é um DAG.

Segundo estudos sobre teoria dos grafos, DAGs são utilizados para representar relações de dependência, causalidade e fluxo de execução em sistemas computacionais.

## 3. Estrutura de um DAG

### 3.1 Nós (Vértices)

Os nós representam entidades, operações ou variáveis.

Exemplos:

- tarefas;
- cálculos matemáticos;
- variáveis;
- funções.

### 3.2 Arestas

As arestas representam conexões entre os nós. Em DAGs, as arestas possuem direção.

Exemplo:

```text
A → B
```

Significa que:

- A influencia B;
- ou B depende de A.

### 3.3 Ausência de Ciclos

A principal característica dos DAGs é a ausência de ciclos.

Isso garante:

- ordem lógica;
- fluxo contínuo;
- execução sem dependências infinitas.

Essa propriedade é extremamente importante em computação, pois evita loops de dependência.

## 4. Ordenação Topológica

A ordenação topológica é um dos conceitos mais importantes relacionados aos DAGs. Ela consiste em organizar os nós em uma sequência válida, respeitando as dependências do grafo.

Se existe `A → B`, então A deve aparecer antes de B.

A ordenação topológica só pode ser aplicada em DAGs. Caso exista um ciclo, não é possível determinar uma ordem válida de execução.

### 4.1 Exemplo de Ordenação Topológica

Considere o seguinte DAG:

```text
A → B
A → C
B → D
C → D
```

Uma possível ordenação seria `A → B → C → D`. Outra possibilidade seria `A → C → B → D`. Ambas são válidas porque respeitam as dependências.

## 5. Aplicações dos DAGs

Os DAGs possuem diversas aplicações na computação.

### 5.1 Sistemas de Dependência

Utilizados para:

- compilação de programas;
- gerenciamento de pacotes;
- execução de tarefas.

Exemplo: uma tarefa só pode iniciar após outra terminar.

### 5.2 Fluxos de Trabalho (Workflows)

DAGs são usados para representar pipelines de processamento.

Exemplo:

- coleta de dados;
- limpeza;
- transformação;
- treinamento;
- avaliação.

Cada etapa depende da anterior.

### 5.3 Bancos de Dados

Sistemas de banco de dados utilizam DAGs para:

- otimização de consultas;
- execução paralela;
- controle de dependências.

### 5.4 Inteligência Artificial

Na IA, DAGs são usados em:

- redes Bayesianas;
- grafos computacionais;
- aprendizado profundo;
- inferência causal.

## 6. DAGs em Machine Learning

Em Machine Learning, DAGs representam relações computacionais entre operações matemáticas.

Frameworks modernos utilizam DAGs para:

- organizar cálculos;
- otimizar execução;
- calcular gradientes;
- executar treinamento de redes neurais.

## 7. Grafos Computacionais

Grafos computacionais são representações matemáticas utilizadas em Deep Learning.

Neles:

- os nós representam operações;
- as arestas representam o fluxo de dados.

Esses grafos geralmente possuem estrutura DAG.

### 7.1 Exemplo de Grafo Computacional

Considere:

$$
y=\operatorname{sigmoid}(wx+b)
$$

O fluxo computacional seria:

```text
Entrada x
    ↓
Multiplicação por w
    ↓
Soma com b
    ↓
Função sigmoid
    ↓
Saída y
```

Cada operação depende da anterior, formando um DAG.

## 8. TensorFlow e DAGs

O TensorFlow utiliza grafos computacionais para representar operações matemáticas.

Segundo a documentação oficial, o TensorFlow usa dataflow graphs para representar:

- computação;
- operações;
- fluxo de dados.

Nesse sistema:

- nós representam operações;
- arestas representam tensores e fluxo de dados.

Essa abordagem permite:

- paralelismo;
- otimização;
- execução distribuída.

## 9. PyTorch e Grafos Dinâmicos

O PyTorch também utiliza DAGs, porém através de grafos dinâmicos.

Diferente do TensorFlow clássico:

- o grafo é construído durante a execução;
- permitindo maior flexibilidade.

Isso facilita:

- depuração;
- desenvolvimento;
- experimentação em Deep Learning.

## 10. Backpropagation e DAGs

Backpropagation é o algoritmo utilizado para treinamento de redes neurais. Ele funciona propagando os gradientes no sentido inverso do grafo computacional.

O processo ocorre em duas etapas:

### 10.1 Forward Pass

Execução das operações até gerar a saída.

### 10.2 Backward Pass

Propagação dos erros para atualizar os pesos.

Como cada operação depende da anterior, o grafo computacional de uma execução é organizado de maneira acíclica para que as dependências sejam percorridas na ordem correta.

## 11. DAGs em Classificação Binária

Na classificação binária, o objetivo é prever duas classes possíveis.

Exemplos:

- spam ou não spam;
- fraude ou não fraude;
- aprovado ou reprovado.

O modelo normalmente segue o fluxo:

```text
Entrada → processamento → função de ativação → saída
```

Esse fluxo pode ser representado como um DAG computacional.

### 11.1 Exemplo de Fluxo

```text
Entrada de dados
    ↓
Camada linear
    ↓
Função sigmoid
    ↓
Probabilidade
    ↓
Classificação final
```

Cada etapa depende da anterior.

## 12. Vantagens dos DAGs

Os DAGs apresentam diversas vantagens:

- organização lógica;
- representação clara de dependências;
- paralelismo;
- otimização computacional;
- facilidade de análise;
- eficiência em Machine Learning.

Além disso, DAGs evitam ciclos infinitos e inconsistências computacionais.

## 13. Desvantagens e Limitações

Apesar das vantagens, DAGs possuem limitações:

- dificuldade em representar sistemas cíclicos;
- complexidade em grafos muito grandes;
- alto custo computacional em alguns algoritmos.

Mesmo assim, continuam sendo fundamentais em IA moderna.

## 14. Aplicações Modernas

Atualmente, DAGs são utilizados em:

- TensorFlow;
- PyTorch;
- Apache Airflow;
- sistemas distribuídos;
- redes Bayesianas;
- pipelines de dados;
- inferência causal;
- métodos de aprendizado de estruturas acíclicas com redes neurais em grafos.

Pesquisas recentes também utilizam DAGs para aprendizado causal e descoberta automática de relações entre variáveis.

## 15. Conclusão

Os DAGs são estruturas fundamentais na computação moderna e possuem grande importância em Machine Learning.

Sua capacidade de representar dependências e fluxos de execução permite organizar operações matemáticas de forma eficiente e otimizada.

Frameworks como TensorFlow e PyTorch utilizam DAGs para construir grafos computacionais responsáveis pelo treinamento de modelos de inteligência artificial.

Além disso, DAGs possuem aplicações em pipelines de dados, workflows, sistemas distribuídos e inferência causal, tornando-se uma das estruturas mais importantes da área de ciência de dados e inteligência artificial.

Com o crescimento do Deep Learning e da IA moderna, os DAGs continuam sendo essenciais para o desenvolvimento de sistemas computacionais eficientes e escaláveis.

## 16. Referências Bibliográficas

### Livros e Artigos

- ABADI, Martín et al. TensorFlow: A System for Large-Scale Machine Learning. OSDI, 2016. Disponível em: <https://research.google/pubs/tensorflow-a-system-for-large-scale-machine-learning/>.
- YU, Yue et al. DAG-GNN: DAG Structure Learning with Graph Neural Networks. 2019. Disponível em: <https://arxiv.org/abs/1904.10098>.
- ZUO, Hao et al. DAGOR: Learning DAGs via Topological Sorts and QR Factorization. Mathematics, 2024. Disponível em: <https://www.mdpi.com/2227-7390/12/8/1198>.

### Sites e Documentações

- TensorFlow Documentation — Computational Graphs. Disponível em: <https://www.tensorflow.org/guide/intro_to_graphs>.
- NetworkX Documentation — Directed Acyclic Graphs. Disponível em: <https://networkx.org/nx-guides/content/algorithms/dag/index.html>.
- GeeksforGeeks — Dynamic vs Static Computational Graphs. Disponível em: <https://www.geeksforgeeks.org/dynamic-vs-static-computational-graphs-pytorch-and-tensorflow/>.
- AI Wiki — Computational Graphs. Disponível em: <https://aiwiki.ai/wiki/computational_graph>.

## Colaboradores

| | |
|:---:|:---:|
| <img loading=lazy src=images/colaboradores/adilson_vissoli.svg width=115><br><sub>Adilson Vissoli</sub> | [<img loading=lazy src=https://avatars.githubusercontent.com/u/207051125?v=4 width=115><br><sub>Arthur Bogoni</sub>](https://github.com/ArthurBogoni) |
