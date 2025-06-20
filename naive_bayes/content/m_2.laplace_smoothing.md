# English Version

# Portuguese version

## **Laplace/Lidstone Smoothing**

### O Problema do Zero

Durante o treinamento do Naive Bayes, pode acontecer de um atributo (como uma palavra) nunca aparecer em uma determinada classe. Por exemplo, imagine que no nosso dataset de e-mails, a palavra "promoção" aparece em 50 e-mails da classe spam, mas em **nenhum** e-mail da classe normal.

**Dados de Treinamento:**
* **Total de E-mails:** 3000 (sendo 900 spam e 2100 normais).
* **Probabilidades a Priori:** $P(\text{spam}) = \frac{900}{3000} = 0.3$ e $P(\text{normal}) = \frac{2100}{3000} = 0.7$.
* **Ocorrência da Palavra "promoção":**
    * A palavra "promoção" aparece 80 vezes nos e-mails spam (de um total de 50.000 palavras)
    * A palavra "promoção" aparece 0 vezes nos e-mails normais (de um total de 120.000 palavras)

Isso resulta no seguinte cálculo de probabilidade:

$P(\text{"promoção"}|\text{spam}) = \frac{80}{50000} \approx 0.0016$

$P(\text{"promoção"}|\text{normal}) = \frac{0}{120000} = 0$

O problema surge quando um novo e-mail chega contendo a palavra "promoção" junto com outras palavras que poderiam indicar que é um email normal. Ao usar log-probabilidades para classificar, teremos:

$$\log P(\text{normal}) + \log P(\text{"promoção"}|\text{normal}) + \log P(\text{outras palavras}|\text{normal})$$

Como $P(\text{"promoção"}|\text{normal}) = 0$, temos $\log(0) = -\infty$, e toda a pontuação para a classe "normal" resulta em $-\infty$. O algoritmo **nunca** classificará um e-mail como normal se ele contiver a palavra "promoção", não importa o que as outras palavras do e-mail sugiram. Esse problema é bastante comum que afeta qualquer aplicação de Naive Bayes. 

### A Solução: Laplace/Lindstone Smoothing 

A solução é simples: adicionamos uma pequena constante $\alpha$ a todas as contagens. Essa técnica é chamada de **smoothing (suavização)**. Em vez de calcular:

$$P(X_i|C) = \frac{\text{Contagem da palavra $X_i$ na classe}}{\text{Nº total de palavras da classe } C}$$

Calculamos:

$$P(X_i|C) = \frac{(\text{Contagem da palavra $X_i$ na classe}) + \alpha}{(\text {Nº total de palavras da classe} C) + \alpha*|\text{vocabulario}|}$$

> [!NOTE]
> **Por que `α × |vocabulário|` no denominador?**
> 
> A intuição é que, para qualquer classe, temos **todas as palavras do vocabulário** como possibilidades. A suavização funciona adicionando `α` "palavras fantasmas" para *cada palavra* do vocabulário, garantindo que nenhuma palavra tenha probabilidade zero.
> * O *`+ α`* no numerador representa as palavras fantasmas adicionadas para a palavra específica que estamos calculando.
> * O `+ α × |vocabulário|` no denominador representa o total de palavras fantasmas adicionadas: `α` para cada uma das `|vocabulário|` palavras possíveis.
> Isso mantém a distribuição de probabilidade válida (somando 1) e torna o modelo matematicamente estável.

**Exemplo prático com smoothing (α = 1):**

$P(\text{"promoção"}|\text{spam}) = \frac{80 + 1}{50.000 + 10.000} = \frac{81}{60.000} = 0.00135$

$P(\text{"promoção"}|\text{normal}) = \frac{0 + 1}{120.000 + 10.000} = \frac{1}{130.000} \approx 0.0000077$

Agora podemos calcular com log-probabilidades sem problemas:

* **SPAM:** $\log(0.3) + \log(0.00135) = -1.204 + (-6.608) = -7.812$
* **NORMAL:** $\log(0.7) + \log(0.0000077) = -0.357 + (-11.772) = -12.129$

Como $-7.812 > -12.129$, classificamos como SPAM. O importante é que agora outras palavras no e-mail podem influenciar a classificação final - o modelo não fica mais "travado" em uma decisão absoluta.

Mesmo que a palavra "promoção" nunca tenha aparecido nos e-mails normais, não é razoável assumir que sua presença elimina completamente a possibilidade de um e-mail ser normal. A suavização reflete essa incerteza: ela assume que mesmo eventos não observados podem acontecer, apenas com baixa probabilidade. Isso evita que o modelo "zere" alternativas e se torne excessivamente rígido.

### Tipos de Smoothing: Lidstone e Laplace

A técnica geral de adicionar uma constante $\alpha$ é chamada de **Lidstone Smoothing**. Ela nos permite controlar a intensidade da suavização:

* **α pequeno (e.g., 0.1-0.5):** Confia mais nos dados originais e aplica uma correção mínima.
* **α grande (>1):** É mais conservador e distribui as probabilidades de forma mais uniforme, suavizando mais os dados.
* O **Laplace Smoothing** (também conhecido como *Add-One Smoothing*) é simplesmente o caso especial e mais famoso do Lidstone Smoothing, onde se escolhe **$\alpha = 1$**. É a escolha padrão em muitas implementações por ser um ponto de partida balanceado e eficaz.

### Base Teórica: MLE vs MAP

A intuição por trás do smoothing tem uma fundamentação teórica sólida. Para entender por que ele funciona, podemos contrastar duas abordagens de estimativa:

- **Estimação por Máxima Verossimilhança (Maximum Likelihood Estimation - MLE): "Só acredito no que vejo nos dados."**
    * Esta abordagem calcula as probabilidades contando simplesmente as ocorrências nos dados de treinamento. O Naive Bayes sem smoothing usa essa filosofia. Sua fórmula é exatamente a que usamos para o cálculo inicial, sem correção:
      
      $$P_{MLE}(X_i \mid C) = \frac{\text{Contagem da palavra } X_i \text{ na classe}}{\text{Número total de palavras da classe } C}$$
  
   * O ponto fraco é que se um evento nunca ocorreu na sua amostra, sua probabilidade é definida como zero, o que torna o modelo frágil.

- **Máximo A Posteriori (MAP): "Eu combino o que eu já acredito com o que os dados me mostram."**
    * A ideia fundamental do MAP é, de fato, **incorporar um "conhecimento prévio" (a priori) aos dados observados** para chegar a uma estimativa final (a posteriori). 
  
    * No nosso contexto, aplicamos este conceito de forma muito pragmática usando **pseudocontagens** (o valor α). Nosso "conhecimento prévio" não é uma crença complexa, mas sim a suposição simples e útil de que **toda palavra tem uma pequena chance de aparecer**, mesmo que não a tenhamos visto.


## Referências
