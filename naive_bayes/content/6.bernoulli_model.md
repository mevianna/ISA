## Modelo Bernoulli

O **Bernoulli Naive Bayes** é ideal para problemas em que os atributos são binários – ou seja, indicam **presença ou ausência** de uma característica, e não a frequência.

> [!NOTE]
>**Diferença do Multinomial**: Enquanto o modelo Multinomial conta **quantas vezes** cada palavra aparece em um documento, o Bernoulli se preocupa apenas com **se a palavra aparece ou não**. Além disso, o Bernoulli considera **tanto a presença quanto a ausência** de cada atributo na classificação — uma palavra que **não aparece** em um documento também influencia a decisão do classificador.

### Exemplo

Vamos supor que temos um vocabulário com 3 palavras: "ganhe", "dinheiro", "agora".

Dois emails:

| Palavra   | Email 1 | Email 2 |
|-----------|---------|---------|
| ganhe     | 1       | 0       |
| dinheiro  | 1       | 1       |
| agora     | 0       | 1       |

Mesmo que "dinheiro" apareça 10 vezes em um email, isso não faz diferença para o Bernoulli Naive Bayes — ele trata como **presente**. Além disso, o fato de "ganhe" **não aparecer** no Email 2 é uma informação relevante para a classificação.

### Cálculo da verossimilhança

Usa-se:

Para cada atributo $X_i$, calculamos as probabilidades de **presença** e **ausência**:

$$P(X_i = 1|C) = \frac{N_{ic} + \alpha}{N_c + 2\alpha}$$

$$P(X_i = 0|C) = 1 - P(X_i = 1|C) = \frac{N_c - N_{ic} + \alpha}{N_c + 2\alpha}$$


- $N_{ic}$ = nº de emails da classe $C$ que contêm a palavra $X_i$
- $N_c$ = total de emails da classe $C$
- $\alpha$ = parâmetro de suavização

> [!NOTE]
> **Por que `2α` no denominador?**
> A intuição é que, para qualquer palavra, existem dois cenários possíveis em um e-mail: a palavra **está presente** ou **não está presente**. A suavização funciona adicionando `α` "e-mails fantasmas" para *cada um* desses cenários, a fim de evitar probabilidades de zero em ambos os casos.
> * O `+ α` no numerador representa o e-mail fantasma em que a palavra **está presente**.
> * O `+ 2α` no denominador representa o total de e-mails fantasmas adicionados: `α` para o cenário "presente" e `α` para o cenário

### Classificador

$$
\hat{C} = \arg\max_C \left[ \log P(C) + \sum_i \big( x_i \log P(X_i=1|C) + (1 - x_i) \log(1 - P(X_i=1|C)) \big) \right]
$$

### O papel do $x_i$ na fórmula

Na equação acima, o termo $x_i$ representa se o atributo  $i$ está **presente (1)** ou **ausente (0)** no exemplo que estamos classificando. Ele atua como um **seletor** que decide qual probabilidade será usada:

- **Se $x_i=1$ (palavra presente)**: o primeiro termo $x_i \log P(X_i=1|C)$ contribui com $\log P(X_i=1|C)$, e o segundo termo $(1-x_i) \log P(X_i=0|C)$ se anula.

- **Se $x_i=0$ (palavra ausente)**: o primeiro termo se anula, e o segundo termo contribui com $\log P(X_i=0|C)$.


**Resultado**: Para cada palavra no vocabulário, o modelo sempre considera uma evidência — seja a probabilidade de presença (quando a palavra aparece) ou a probabilidade de ausência (quando não aparece).

> [!TIP]
> Bernoulli NB pode ser mais adequado que o Multinomial quando os documentos são curtos ou quando a **frequência** de palavras é irrelevante.
