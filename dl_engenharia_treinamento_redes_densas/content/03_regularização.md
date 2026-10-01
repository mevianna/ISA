## Overfitting (Sobreajuste)

O *overfitting* ocorre quando a rede cria regras complexas demais, memorizando o ruído e as anomalias exclusivas dos dados de treinamento, em vez de capturar o padrão geral. O modelo acerta tudo no treino, mas falha gravemente no mundo real. O objetivo principal do treinamento de uma inteligência artificial é garantir a melhoria da generalização, que é justamente a capacidade do modelo de manter um alto desempenho em dados que ele nunca viu antes.

## Técnicas de Regularização

Para resolver o problema do sobreajuste, aplicamos a Regularização. A regularização atua "puxando o freio" do modelo, adicionando restrições matemáticas e estruturais durante o treinamento para impedir que a rede decore os dados. O foco deste tópico é o uso de *Dropout* e penalidades L2 (*Weight Decay*) para mitigação de *overfitting* e melhoria da generalização.

### 1. Penalidade L2 (Weight Decay)

A técnica de penalidades L2, também muito conhecida na literatura como *Weight Decay* (decaimento de pesos), age diretamente na função matemática de erro (Função de Custo) da rede neural.

**A Fórmula Matemática:**

Em uma rede sem regularização, a função tenta apenas minimizar o erro das previsões. Com a L2, a nova função de custo passa a ser o erro original somado a um termo de penalidade:

$$ \text{Custo Total} = \text{Custo Original} + \frac{\lambda}{2} \sum w^2 $$

Onde:

- **Custo Original:** É o erro padrão da rede.
- **$\lambda$ (Lambda):** É o hiperparâmetro de regularização. Ele define a "força" da punição.
- **$\sum w^2$ (Soma dos pesos ao quadrado):** É o valor que penaliza a rede caso os pesos fiquem muito grandes.

**Pergunta importante: Por que penalizar pesos grandes?** 
Na dinâmica de aprendizado das redes neurais, pesos matemáticos muito grandes significam que o modelo está dando uma importância desproporcional a uma única característica dos dados. Ao forçar a rede a manter seus pesos baixos e bem distribuídos, o *Weight Decay* torna o modelo mais equilibrado, considerando todas as informações de forma suave. Isso é essencial para a mitigação de *overfitting*.

Além disso, a penalização L2 não impede que a rede utilize determinadas características com maior importância. O objetivo é evitar que os pesos assumam valores excessivamente grandes, restringindo a complexidade da solução encontrada pelo modelo.

#### Como a L2 afeta o treinamento?

Durante o treinamento, a rede utiliza algoritmos como o Gradiente Descendente para atualizar seus pesos.

Sem regularização, a atualização de um peso pode ser representada simplificadamente por:

$$ w_{\text{novo}} = w_{\text{antigo}} - \eta \times \text{Gradiente} $$

Com a penalização L2, o gradiente passa a considerar também o termo de regularização:

$$ \text{Gradiente Total} = \text{Gradiente do Erro} + \lambda w $$

Assim, a atualização passa a possuir uma tendência adicional de reduzir o valor dos pesos.

Matematicamente:

$$ w \leftarrow w - \eta \left( \frac{\partial C}{\partial w} + \lambda w \right) $$

Onde:

- **$w$:** peso da rede;
- **$\eta$:** taxa de aprendizado;
- **$\frac{\partial C}{\partial w}$:** gradiente do custo original em relação ao peso;
- **$\lambda w$:** contribuição da regularização L2.

Isso explica a ideia de *Weight Decay*: além de tentar diminuir o erro das previsões, o treinamento também possui uma tendência de manter os pesos menores.

#### Efeito do valor de Lambda ($\lambda$)

O valor de Lambda ($\lambda$) é extremamente importante.

Se o Lambda for muito pequeno, a penalização será fraca:

> **Lambda baixo $\rightarrow$ pouca regularização $\rightarrow$ maior risco de *overfitting***

Se o Lambda for muito alto, a penalização será muito forte:

> **Lambda alto $\rightarrow$ regularização excessiva $\rightarrow$ possível perda de capacidade de aprendizado**

Portanto, existe um equilíbrio que precisa ser encontrado durante o treinamento.

Um Lambda excessivamente alto pode fazer com que a rede não consiga aprender nem mesmo os padrões importantes presentes nos dados, levando a um modelo excessivamente simples.

Por isso, o valor de Lambda é um **hiperparâmetro**, e normalmente é escolhido através de experimentação e avaliação utilizando um conjunto de validação.

---

### 2. Dropout (Descarte Aleatório)

Enquanto o *Weight Decay* atua na matemática do erro, o uso de *Dropout* atua na estrutura da arquitetura da rede durante o seu treinamento.

**Como Funciona:**

Em cada rodada do treinamento, o algoritmo "desliga" (passa a valer zero) uma porcentagem aleatória de neurônios de uma camada. Geralmente, essa taxa de descarte varia entre 20% e 50%. A fórmula simples para a saída do neurônio durante o treino fica:

**Saída = 0 (com probabilidade de descarte $p$) ou Saída Original (caso o neurônio seja mantido).**

O ponto fundamental é que os neurônios desligados são escolhidos **aleatoriamente a cada etapa do treinamento**. Portanto, a configuração da rede utilizada em uma etapa pode ser diferente da configuração utilizada na etapa seguinte.

Por exemplo, considere uma camada com 10 neurônios e um *Dropout* de 30%.

Em uma determinada etapa, aproximadamente 3 neurônios poderão ser desativados:
**10 neurônios $\rightarrow$ 7 ativos + 3 desativados**

Na próxima etapa, outros neurônios podem ser selecionados:
**10 neurônios $\rightarrow$ 6 ativos + 4 desativados**

O conjunto exato depende do sorteio realizado durante o treinamento.

#### Por que desligar neurônios ajuda contra o Overfitting?

Sem *Dropout*, determinados neurônios podem acabar se tornando excessivamente dependentes de outros neurônios específicos. Isso pode fazer com que a rede aprenda combinações muito específicas dos dados de treinamento.

O *Dropout* dificulta esse comportamento porque a rede precisa continuar funcionando mesmo quando alguns dos seus neurônios não estão disponíveis. Assim, a informação precisa ser distribuída entre diferentes neurônios.

A ideia pode ser resumida como:

**Sem Dropout:**
> "Eu posso depender desses neurônios específicos para realizar esta previsão."

**Com Dropout:**
> "Preciso aprender uma representação que continue funcionando mesmo que alguns neurônios sejam removidos."

Isso reduz a dependência excessiva entre neurônios e pode melhorar a capacidade de generalização.

#### Probabilidade de Dropout

A taxa de *Dropout* é representada normalmente por $p$, que representa a probabilabilidade de um neurônio ser descartado.

Por exemplo:

- **$p = 0.1 \rightarrow$ 10% dos neurônios são descartados**
- **$p = 0.2 \rightarrow$ 20%**
- **$p = 0.3 \rightarrow$ 30%**
- **$p = 0.5 \rightarrow$ 50%**

Quanto maior o valor de $p$, maior será a regularização aplicada à rede.

Porém, assim como ocorre com o Lambda na regularização L2, um valor excessivamente alto pode prejudicar o aprendizado. Um *Dropout* muito forte pode fazer com que informações importantes sejam constantemente removidas, dificultando o treinamento.

---

## Dropout durante o treinamento e durante a utilização do modelo

Um ponto importante é que o *Dropout* é utilizado principalmente **durante o treinamento**.

- **Durante o treinamento:** Alguns neurônios $\rightarrow$ desligados aleatoriamente.
- **Durante a utilização (inferência):** Todos os neurônios $\rightarrow$ disponíveis.

Ou seja, o comportamento utilizado para treinar a rede não é exatamente o mesmo utilizado durante a inferência. Para compensar a diferença na quantidade de neurônios ativos, utiliza-se normalmente o chamado **Inverted Dropout**.

Nesse método, os valores dos neurônios que permanecem ativos são escalados durante o treinamento.

Se a probabilidade de descarte for $p$, a probabilidade de um neurônio permanecer ativo será: $1-p$

A saída de um neurônio mantido pode ser escalada por: $\frac{1}{1-p}$

Assim, a magnitude média das ativações permanece aproximadamente consistente entre treinamento e inferência.

Por exemplo, com $p=0.5$, a probabilidade de um neurônio permanecer ativo é $1-p=0.5$.

Nesse caso, as ativações mantidas são multiplicadas por: $\frac{1}{0.5}=2$

Isso permite que, durante a inferência, todos os neurônios possam ser utilizados sem a necessidade de realizar um novo descarte aleatório.

---

## Relação entre Dropout e Overfitting

O *Dropout* combate o *overfitting* principalmente ao impedir que a rede dependa excessivamente de determinados neurônios durante o treinamento.

A lógica é:

**Dropout**
$\downarrow$
**Neurônios são aleatoriamente desativados**
$\downarrow$
**A rede não pode depender excessivamente de neurônios específicos**
$\downarrow$
**As informações precisam ser distribuídas**
$\downarrow$
**Menor tendência à memorização dos dados de treinamento**
$\downarrow$
**Melhor generalização**

Portanto, o objetivo não é simplesmente "desligar neurônios", mas fazer com que a rede aprenda representações mais robustas.

---

## Uso combinado de L2 e Dropout

As duas técnicas não são necessariamente alternativas. É possível utilizar **L2 e Dropout simultaneamente** na mesma rede.

Nesse caso, cada técnica atua sobre um aspecto diferente do treinamento:

- **L2 $\rightarrow$** controla a magnitude dos pesos.
- **Dropout $\rightarrow$** reduz a dependência excessiva entre neurônios.

A combinação pode ser útil quando uma única técnica não é suficiente para controlar o *overfitting*.

Porém, aplicar regularização em excesso também pode prejudicar o modelo. Se tanto o Lambda quanto a taxa de *Dropout* forem muito altos, a rede poderá apresentar dificuldade para aprender os padrões importantes dos dados.

Por isso, esses valores devem ser tratados como **hiperparâmetros** e ajustados experimentalmente.

## Colaboradores

| | |
|:---:|:---:|
| <img loading="lazy" src="images/colaboradores/lucas_schemes.svg" width="115"><br><sub>Lucas Schemes</sub> | <img loading="lazy" src="images/colaboradores/daiana_brum.svg" width="115"><br><sub>Daiana Brum</sub> |
