## Modelo Bernoulli

O **Bernoulli Naive Bayes** é ideal para problemas em que os atributos são binários – ou seja, indicam **presença ou ausência** de uma característica, e não a frequência.

> [!NOTE]
> No contexto de texto, por exemplo, esse modelo não considera quantas vezes uma palavra aparece em um documento, mas apenas **se ela aparece ou não**.

### Exemplo

Vamos supor que temos um vocabulário com 3 palavras: "ganhe", "dinheiro", "agora".

Dois emails:

| Palavra   | Email 1 | Email 2 |
|-----------|---------|---------|
| ganhe     | 1       | 0       |
| dinheiro  | 1       | 1       |
| agora     | 0       | 1       |

Mesmo que "dinheiro" apareça 10 vezes em um email, isso não faz diferença para o Bernoulli Naive Bayes — ele trata como **presente**.

### Cálculo da verossimilhança

Usa-se:

$$
P(X_i = 1|C) = \frac{N_{ic} + \alpha}{N_c + 2\alpha}
$$

$$
P(X_i = 0|C) = 1 - P(X_i = 1|C)
$$

- $N_{ic}$ = nº de documentos da classe $C$ que contêm a palavra $X_i$
- $N_c$ = total de documentos da classe $C$
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

Na equação acima, o termo $ x_i$ representa se o atributo  $i$ está **presente (1)** ou **ausente (0)** no exemplo que estamos classificando. Ele atua como um **seletor** que decide qual probabilidade será usada:

- Se $ x_i=1$ , o termo se reduz a $( \log P(X_i = 1 | C) )$, ou seja, usamos a **probabilidade de presença**.
- Se $ x_i=0$, o termo vira $( \log(1 - P(X_i = 1 | C)) )$, ou seja, usamos a **probabilidade de ausência**.

Essa estrutura permite que o modelo Bernoulli leve em consideração **não apenas as palavras que estão presentes no documento**, mas também aquelas que estão **ausentes**, o que é uma grande diferença em relação ao modelo Multinomial. Isso faz com que o Bernoulli NB seja mais sensível ao padrão geral de presença e ausência de termos, o que pode ser útil especialmente em documentos curtos ou filtrados.

> [!TIP]
> Bernoulli NB pode ser mais adequado que o Multinomial quando os documentos são curtos ou quando a **frequência** de palavras é irrelevante.
