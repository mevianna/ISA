

# Técnicas de inicialização de parâmetros (Xavier/He)

## Introdução: por que a inicialização importa

Antes do treino, os pesos de uma rede precisam receber algum valor inicial. Se a escala for inadequada, o sinal se deteriora ao atravessar as camadas:

- **Pesos muito pequenos:** a variância das ativações encolhe a cada camada, o sinal tende a zero e os gradientes desaparecem (*vanishing gradients*).
- **Pesos muito grandes:** a variância cresce a cada camada, os gradientes explodem (*exploding gradients*) e ativações como tanh/sigmoide saturam, travando o aprendizado.

> [!NOTE]
> Pense em uma fila de pessoas passando uma mensagem adiante: se cada uma sussurra baixo demais, a mensagem some; se cada uma grita, ela vira ruído. A inicialização busca o volume certo para que a mensagem chegue ao fim com a mesma intensidade com que começou.

A ideia central das duas técnicas é a mesma: **escolher a variância dos pesos de modo que a variância das ativações (propagação direta) e dos gradientes (retropropagação) se mantenha aproximadamente constante de camada para camada.** Elas diferem na função de ativação para a qual a conta é feita [4] [5].

---

## Inicialização de Xavier (Glorot)

Proposta por Glorot e Bengio (2010) [4].

### Derivação

Considere uma camada linear $y = \sum_{i=1}^{n_{in}} w_i x_i$, com pesos e entradas independentes e de média zero. Então:

$$\mathrm{Var}(y) = n_{in}\,\mathrm{Var}(w)\,\mathrm{Var}(x)$$

- **Propagação direta (forward):** para manter $\mathrm{Var}(y)=\mathrm{Var}(x)$, é preciso $\mathrm{Var}(w) = \dfrac{1}{n_{in}}$.
- **Retropropagação (backward):** o mesmo raciocínio aplicado aos gradientes leva a $\mathrm{Var}(w) = \dfrac{1}{n_{out}}$.

As duas condições só coincidem quando $n_{in}=n_{out}$. Glorot e Bengio adotaram um compromisso entre elas:

$$\mathrm{Var}(w) = \frac{2}{n_{in}+n_{out}}$$

### Implementação

| Variante | Distribuição |
|---|---|
| Normal | $W \sim \mathcal{N}\!\left(0,\ \dfrac{2}{n_{in}+n_{out}}\right)$ |
| Uniforme | $W \sim \mathcal{U}\!\left[-\sqrt{\dfrac{6}{n_{in}+n_{out}}},\ +\sqrt{\dfrac{6}{n_{in}+n_{out}}}\right]$ |

> [!IMPORTANT]
> O limite $\sqrt{6/(n_{in}+n_{out})}$ vem do fato de a variância de uma distribuição uniforme $\mathcal{U}[-a,a]$ ser $a^2/3$.

### Hipóteses e limitações

- Assume ativação aproximadamente **linear em torno de zero** (tanh e softsign satisfazem essa condição).
- **Não** considera que a ReLU zera metade das ativações. Em redes ReLU profundas, a variância encolhe a cada camada, o que motivou a proposta de He et al. [5].

**Quando usar:** tanh, sigmoide, softsign, ativações aproximadamente lineares e camadas de saída em geral.

---

## Inicialização de He (Kaiming)

Proposta por He et al. (2015) [5], no mesmo artigo que introduz a ativação PReLU.

### Derivação

Para a camada $l$, com $y_l = W_l x_l + b_l$ e $x_l = \max(0, y_{l-1})$:

$$\mathrm{Var}(y_l) = n_l\,\mathrm{Var}(w_l)\,\mathbb{E}[x_l^2]$$

Aqui aparece $\mathbb{E}[x_l^2]$ e não $\mathrm{Var}(x_l)$, porque a saída da ReLU **não tem média zero**. Se $y_{l-1}$ é simétrica em torno de zero, a ReLU elimina metade da massa, e portanto:

$$\mathbb{E}[x_l^2] = \tfrac{1}{2}\,\mathrm{Var}(y_{l-1})$$

Logo:

$$\mathrm{Var}(y_l) = \tfrac{1}{2}\, n_l\,\mathrm{Var}(w_l)\,\mathrm{Var}(y_{l-1})$$

Ao empilhar $L$ camadas, surge o produto $\prod_l \tfrac{1}{2} n_l \mathrm{Var}(w_l)$. Para que ele não exploda nem desapareça, cada fator deve valer 1:

$$\mathrm{Var}(w) = \frac{2}{n_{in}}$$

> [!NOTE]
> O **fator 2** compensa exatamente a metade das ativações eliminada pela ReLU. Na prática, é a inicialização de Xavier (versão *fan-in*) com o dobro da variância.

### Fan-in vs. fan-out

- **`fan_in`** ($n_{in}$): preserva a variância no forward (padrão usual).
- **`fan_out`** ($n_{out}$): preserva a variância dos gradientes no backward.

O artigo original argumenta que qualquer um dos dois é suficiente [5].

### Implementação

| Variante | Distribuição |
|---|---|
| Normal | $W \sim \mathcal{N}\!\left(0,\ \dfrac{2}{n_{in}}\right)$ |
| Uniforme | $W \sim \mathcal{U}\!\left[-\sqrt{\dfrac{6}{n_{in}}},\ +\sqrt{\dfrac{6}{n_{in}}}\right]$ |

### Generalização para Leaky ReLU / PReLU

Com inclinação negativa $a$:

$$\mathrm{Var}(w) = \frac{2}{(1+a^2)\,n_{in}}$$

Com $a=0$ recupera-se a ReLU.

### Resultado empírico

No artigo original, em uma rede de 30 camadas a inicialização de Xavier estagnou, enquanto a de He convergiu. Em redes de cerca de 22 camadas ambas convergiram, mas He o fez mais rápido [5].

**Quando usar:** ReLU, Leaky ReLU, PReLU e variantes.

---

## Comparação

| | **Xavier/Glorot** | **He/Kaiming** |
|---|---|---|
| Ano | 2010 | 2015 |
| Ativação-alvo | tanh, sigmoide, linear | ReLU e derivadas |
| Variância | $\dfrac{2}{n_{in}+n_{out}}$ | $\dfrac{2}{n_{in}}$ (ou $\dfrac{2}{n_{out}}$) |
| Usa fan_in e fan_out | Ambos | Um ou outro |
| Hipótese-chave | Ativação linear, média zero | ReLU zera metade das ativações |
| Falha típica | Variância encolhe em redes ReLU profundas | Pode saturar com tanh/sigmoide |

Existe ainda a inicialização de **LeCun**, $\mathrm{Var}(w)=1/n_{in}$ [6], usada com a ativação SELU em redes auto-normalizantes.

---

## Uso prático

### PyTorch

```python
import torch.nn as nn

# Xavier
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

# He (Kaiming)
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_uniform_(layer.weight, a=0.01, nonlinearity='leaky_relu')

nn.init.zeros_(layer.bias)
```

### Keras / TensorFlow

```python
from tensorflow.keras import layers, initializers

layers.Dense(128, activation='relu',
             kernel_initializer=initializers.HeNormal())
layers.Dense(128, activation='tanh',
             kernel_initializer=initializers.GlorotUniform())  # padrão do Keras
```

### Boas práticas

- Inicialize os **vieses (biases) com zero**.
- Combine a inicialização com a ativação: ReLU → He; tanh/sigmoide → Xavier.
- Os pesos **não** devem ser todos iguais nem todos zero, pois isso impede a quebra de simetria.

---

#  Batch Normalization

## Introdução e problema do Internal Covariate Shift (Desvio de Covariância Interno)

Ao treinar redes neurais profundas, uma das principais problemáticas é a desordem nas camadas internas. A distribuição das entradas de cada camada se altera continuamente à medida que os parâmetros (pesos e vieses)
das camadas anteriores são atualizados durante a otimização. Esse fenômeno, conhecido como Internal Covariate Shift (Desvio de Covariância Interno), faz com que pequenas alterações iniciais se amplifiquem à medida
que a rede se aprofunda, forçando as camadas posteriores a se adaptarem constantemente a novas distribuições estatísticas [1]. 

> [!NOTE]
> Na prática, é como se você estivesse tentando construir uma torre de blocos onde o chão de cada andar mexe sem parar enquanto você coloca os blocos de cima.
> Isso deixa o treino muito lento, exige que o programador use taxas de aprendizagem microscópicas e torna o modelo super sensível na hora de inicializar os números.
>

## O Mecanismo do Batch Normalization

O princípio central do Batch Normalization é inserir uma camada de normalização logo antes das funções de ativação. Devido ao enorme custo computacional que teria, em vez de calcular estatísticas globais de todo o conjunto de dados,
a técnica normaliza cada recurso escalar de forma independente utilizando a média e a variância calculadas no mini-lote (mini-batch) corrente [1].

> [!IMPORTANT]
> Um mini-batch é um subconjunto do conjunto de treinamento usado durante o processo de aprendizado de uma rede neural.

No tópico *Normalization via Mini-Batch Statistics* Ioffe et al. (2015) descrevem matematicamente como funciona o processo de Batch Normalization. Vamos destrinchar os passos principais das fórmulas:

### Cálculo da média do mini-batch
Centraliza os dados em torno de zero. Onde 𝜇_𝐵 é a média das ativações 𝑥_𝑖 dentro de um mini-batch de tamanho 𝑚.

<img width="191" height="116" alt="image" src="https://github.com/user-attachments/assets/ff11f165-75e4-4ff3-9499-2977f0f13a77" />

### Cálculo da variância do mini-batch
Mede a dispersão das ativações em relação à média. É usada para ajustar a escala dos dados.

<img width="260" height="107" alt="image" src="https://github.com/user-attachments/assets/b1e5b90d-be57-40e4-8640-df4bf902a09e" />


### Normalização das ativações
Cada ativação é normalizada para ter média 0 e variância 1. O termo 𝜖 é um pequeno valor constante para evitar divisão por zero.

<img width="162" height="110" alt="image" src="https://github.com/user-attachments/assets/262a0e54-7a7c-4ab4-a9d3-db85b3cf6783" />


### Transformação afim (aprendida pela rede)
Para garantir que essa normalização não destrua a capacidade expressiva da rede, o algoritmo introduz parâmetros de escala ($\gamma$) e deslocamento ($\beta$) que são aprendidos 
conjuntamente pela rede via retropropagação, 
permitindo que o modelo desfaça a normalização caso isso seja o ideal para o aprendizado.

<img width="158" height="70" alt="image" src="https://github.com/user-attachments/assets/1b1d2f53-5c12-408d-9fbe-60a23e9a6c3f" />

## Treino vs. Inferência
O comportamento do Batch Normalization muda entre as fases de desenvolvimento e produção. Durante o treino, o modelo utiliza as médias e variâncias estipuladas dinamicamente por cada mini-lote (mini-batch), 
aproveitando o dinamismo para guiar a otimização e suavizar o espaço de perda [1] [3]. Já durante a inferência, o comportamento estocástico é totalmente congelado. O sistema passa a utilizar médias populacionais globais acumuladas ao longo do treinamento, garantindo que o modelo seja determinístico, 
rápido e produza exatamente a mesma resposta para um mesmo dado de entrada [1].

## Análises além do Internal Covariate Shift (Desvio de Covariância Interno)
Embora grande parte desse arquivo tenha sido estruturado com base no artigo de Ioffe et al. (2015), onde é atribuído o sucesso do Batch Normalization à redução do Internal Covariate Shift, 
outras duas pesquisas feitas posteriormente [2] e [3] vieram aprofundar e contestar essa visão,
demonstrando que os benefícios reais da técnica passam por outros mecanismos de otimização:

- **Ajuste de Taxas de Aprendizagem e Generalização**: Bjorck et al. (2018) demonstraram na prática que o principal benefício do Batch Normalization é permitir a utilização de taxas
de aprendizagem muito mais altas. Redes sem normalização divergem rapidamente com taxas altas porque pequenas atualizações geram uma explosão incontrolável nas ativações ao longo da profundidade da rede.
Ao forçar as ativações a terem média zero e desvio padrão unitário, o Batch Normalization age como uma "precaução de segurança" contra essa explosão numérica.
Além disso, taxas de aprendizagem maiores geram um ruído benéfico na descida de gradiente estocástica (SGD), o que evita que o modelo fique preso em mínimos locais acentuados e melhora a generalização [2].

- **Suavização do Espaço de Perda (Loss Landscape)**: Santurkar et al. (2018) provaram que o principal impacto do Batch Normalization não está ligado ao controle do Internal Covariate Shift
(mostrando inclusive que redes com BN ainda podem apresentar variações distributivas). O verdadeiro superpoder do Batch Normalization é tornar a paisagem da função de perda (loss landscape) significativamente mais suave.
Essa suavização reduz os valores da constante de Lipschitz da perda e dos gradientes,
tornando o comportamento da otimização muito mais previsível, estável e imune a gradientes extremos [3].

## Referências
[1] Ioffe, S. &amp; Szegedy, C.. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. Proceedings of the 32nd International Conference on Machine Learning, 
in Proceedings of Machine Learning Research 37:448-456 Available from https://proceedings.mlr.press/v37/ioffe15.html.

[2] Bjorck, N., Gomes, C., Selman, B., & Weinberger, K. (2018). *Understanding batch normalization*. Advances in neural information processing systems, 31. 
Available from https://proceedings.neurips.cc/paper_files/paper/2018/hash/36072923bfc3cf47745d704feb489480-Abstract.html

[3] Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). *How does batch normalization help optimization?*. Advances in neural information processing systems, 31.
Available from https://proceedings.neurips.cc/paper_files/paper/2018/hash/905056c1ac1dad141560467e0a99e1cf-Abstract.html

[4] Ioffe, S. & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. Proceedings of the 32nd International Conference on Machine Learning, in Proceedings of Machine Learning Research 37:448-456. Disponível em https://proceedings.mlr.press/v37/ioffe15.html.

[5] Bjorck, N., Gomes, C., Selman, B., & Weinberger, K. (2018). *Understanding batch normalization*. Advances in Neural Information Processing Systems, 31. Disponível em https://proceedings.neurips.cc/paper_files/paper/2018/hash/36072923bfc3cf47745d704feb489480-Abstract.html

[6] Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). *How does batch normalization help optimization?*. Advances in Neural Information Processing Systems, 31. Disponível em https://proceedings.neurips.cc/paper_files/paper/2018/hash/905056c1ac1dad141560467e0a99e1cf-Abstract.html

[7] Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 9:249-256. Disponível em https://proceedings.mlr.press/v9/glorot10a.html


## Colaboradores

| |
|:---:|
| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/112569754?v=4" width="115"><br><sub>Alice Motin Bastos </sub>](https://github.com/AliceMotin) |


| |
|:---:|
| [<img loading="lazy" src="https://github.com/ArthurBogoni.png" width="115"><br><sub>Arthur Bogoni</sub>](https://github.com/ArthurBogoni) |
