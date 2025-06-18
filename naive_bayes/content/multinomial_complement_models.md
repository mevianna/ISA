# English Version

# Portuguese Version

## **Modelos Probabilísticos no Naive Bayes**

Até agora, nossos exemplos de classificação de emails usaram implicitamente a distribuição multinomial. Esta é apenas uma das várias distribuições de probabilidade que podemos usar com Naive Bayes, cada uma adequada para diferentes tipos de dados e problemas.

> [!NOTE]
> **O que é uma Distribuição de Probabilidade?**
> 
> Uma distribuição de probabilidade é uma função matemática que descreve a probabilidade de ocorrência de diferentes resultados possíveis para um experimento.
>
> No Naive Bayes, a "distribuição" é a nossa **suposição** sobre como os dados (atributos) se comportam dentro de cada classe. A escolha da distribuição define como calculamos a verossimilhança $P(X|C)$:
> - Se os atributos são **contagens** (como palavras em um texto), usamos a **Multinomial**.
> - Se os atributos são de **presença/ausência** (sim/não), usamos a **Bernoulli**.
> - Se os atributos são **contínuos** (como altura ou preço), usamos a **Gaussiana**.
>
> Escolher o modelo correto é fundamental para o sucesso do classificador.

Vamos começar formalizando o modelo que temos usado, a distribuição Multinomial, antes de apresentar suas variações e outras distribuições.

### Modelo Multinomial

O Multinomial Naive Bayes é ideal para atributos que representam contagens ou frequências (inteiros não negativos). No contexto de texto, cada documento é um conjunto de atributos, onde cada atributo $X_i$ é a contagem da i-ésima palavra do vocabulário.

#### Teorema de Bayes para o Modelo Multinomial
A verossimilhança $P(X|C)$ é calculada com base na distribuição multinomial. A probabilidade de um documento $d$ pertencer à classe $C$ é dada pela frequência relativa das palavras do documento naquela classe.

#### Classificador Multinomial Naive Bayes
A fórmula que derivamos na introdução é, na verdade, a do classificador Multinomial:

$$\hat{C} = \arg\max_C \left[ \log P(C) + \sum_{i=1}^{n} \log P(X_i|C) \right]$$

Onde $P(X_i|C)$ é a probabilidade da palavra $X_i$ ocorrer na classe $C$, calculada como:

$$P(X_i|C) = \frac{\text{Contagem da palavra } X_i \text{ na classe } C}{\text{Nº total de palavras na classe } C}$$

#### Suavização
Como vimos, este modelo depende diretamente do **Laplace/Lidstone Smoothing** para lidar com palavras que não aparecem em uma determinada classe durante o treinamento (o problema do zero).

$$P(X_i|C) = \frac{(\text{Contagem da palavra } X_i \text{ na classe } C) + \alpha}{(\text{Nº total de palavras na classe } C) + \alpha \cdot |\text{vocabulário}|}$$

### Complement Naive Bayes (CNB)

Complement Naive Bayes (CNB) é uma adaptação do modelo Multinomial, projetado para lidar melhor com datasets desbalanceados. Em datasets desbalanceados, o Multinomial NB tende a favorecer classes majoritárias, pois elas têm maior probabilidade a priori $P(C)$. O CNB contorna isso focando em quão bem cada documento se distingue do **complemento** de cada classe.

Em vez de calcular a probabilidade de um documento pertencer à classe $C$, ele calcula quão incompatível o documento é com o **complemento** de $C$ (todas as outras classes) e escolhe a classe que é mais incompatível com seu complemento.

#### Classificador Complement Naive Bayes

A decisão é baseada em encontrar a classe que **minimiza** a seguinte expressão:

$$\hat{C} = \arg\min_C \sum_{i=1}^{n} \log P(X_i|\bar{C})$$

A estimativa de $P(X_i|\bar{C})$ segue a mesma lógica da verossimilhança multinomial, mas as contagens são feitas em todas as classes, *exceto* a classe $C$:

$$P(X_i|\bar{C}) = \frac{(\text{Contagem da palavra } X_i \text{ no complemento de } C) + \alpha}{(\text{Nº total de palavras no complemento de } C) + \alpha \cdot |\text{vocabulário}|}$$

Mais formalmente:

$$P(X_i|\bar{C}) = \frac{\alpha + \sum_{c \neq C} N_{ic}}{\alpha |\text{vocabulário}| + \sum_{c \neq C} \sum_{j=1}^{|\text{vocabulário}|} N_{jc}}$$

onde $N_{ic}$ é a contagem da palavra $i$ na classe $c$, e $\alpha$ é o parâmetro de suavização.

#### Exemplo Prático: Classificação de Críticas de Cinema

Vamos usar o cenário com três classes: **Comédia**, **Ação** e **Drama**. A classe "Comédia" é a minoritária.

Recebemos a nova crítica: **"piada hilária e divertida"**.

O CNB calculará a **log-probabilidade do complemento** para cada classe. Este valor representa o quão provável é que a crítica pertença ao conjunto de classes do complemento.

Vamos supor que, após os cálculos, o algoritmo chegou aos seguintes valores:

* **Teste para "Comédia"**: (Calcula a log-probabilidade da crítica pertencer a {Ação, Drama})
    * $\sum_{i=1}^{n} \log P(X_i|\overline{\text{Comédia}})$ = **-25.8**

* **Teste para "Ação"**: (Calcula a log-probabilidade da crítica pertencer a {Comédia, Drama})
    * $\sum_{i=1}^{n} \log P(X_i|\overline{\text{Ação}})$ = **-12.5**

* **Teste para "Drama"**: (Calcula a log-probabilidade da crítica pertencer a {Comédia, Ação})
    * $\sum_{i=1}^{n} \log P(X_i|\overline{\text{Drama}})$ = **-14.2**

**Decisão:**

O classificador usa $\arg\min$, ou seja, ele procura a classe que resultou na **menor log-probabilidade do complemento**.

Comparando os resultados: $-25.8$ é o menor valor.

O valor $-25.8$ para Comédia indica que "piada hilária e divertida" é **muito incompatível** com o conjunto {Ação, Drama}. Quanto menor (mais negativo) esse valor, maior a incompatibilidade com o complemento, logo maior a probabilidade de pertencer à classe testada.

Portanto, o modelo classifica a crítica como **Comédia**. Ele conclui que a crítica é tão incompatível com o conjunto {Ação, Drama} que ela muito provavelmente pertence à classe "Comédia".

#### Vantagem em Datasets Desbalanceados

Esta abordagem é especialmente eficaz em cenários desbalanceados porque, ao focar no complemento, o algoritmo dá peso similar a todas as classes, independentemente de sua frequência no dataset de treino. Isso permite que classes minoritárias tenham uma chance mais justa de serem selecionadas quando realmente apropriado.
## Referências
