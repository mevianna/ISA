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

## Referências
