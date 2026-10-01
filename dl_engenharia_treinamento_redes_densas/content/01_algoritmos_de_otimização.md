# Algoritmos de Otimização em Redes Neurais Densas

## Introdução

O treinamento de uma rede neural consiste em ajustar seus parâmetros para reduzir o erro das previsões realizadas pelo modelo. Para isso, algoritmos de otimização utilizam os gradientes calculados durante a retropropagação para definir como os pesos e vieses devem ser atualizados. Entre os principais métodos utilizados em redes neurais densas estão o SGD, o SGD com Momentum, o RMSProp e o Adam. Embora tenham o mesmo objetivo, cada algoritmo realiza essas atualizações de uma maneira diferente, o que pode influenciar a velocidade e a estabilidade do treinamento.

## Estrutura das redes densas

Uma rede neural densa, também chamada de rede totalmente conectada, é composta por uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. Em cada camada, os valores recebidos são multiplicados por pesos, somados a um viés e submetidos a uma função de ativação.

Para uma camada $l$, a combinação linear pode ser representada por:

$$
z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
$$

A saída da camada é então calculada por:

$$
a^{(l)} = f\left(z^{(l)}\right)
$$

Nessas expressões:

- $W^{(l)}$ representa a matriz de pesos da camada;
- $b^{(l)}$ representa o vetor de vieses;
- $a^{(l-1)}$ corresponde à saída da camada anterior;
- $f$ é a função de ativação;
- $a^{(l)}$ é a saída da camada atual.

Durante o treinamento, a rede compara suas previsões com os valores esperados por meio de uma função de perda. Em seguida, o algoritmo de retropropagação calcula os gradientes da perda em relação aos pesos e vieses. O otimizador utiliza essas informações para determinar a direção e o tamanho das atualizações dos parâmetros.

## SGD puro

O método de Descida do Gradiente Estocástica, conhecido como SGD (*Stochastic Gradient Descent*), atualiza os parâmetros utilizando o gradiente calculado a partir de um mini-batch de exemplos. A regra básica de atualização é:

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

Em que:

- $\theta_t$ representa os parâmetros no passo $t$;
- $\eta$ é a taxa de aprendizado;
- $g_t$ é o gradiente da função de perda no passo atual.

O SGD é considerado um método simples porque utiliza diretamente o gradiente atual para modificar os parâmetros. Essa característica facilita sua compreensão e reduz o uso de memória durante o treinamento.

Entretanto, o método apresenta algumas limitações. Como a mesma taxa de aprendizado é aplicada a todos os parâmetros, pesos de diferentes camadas podem receber atualizações inadequadas. Além disso, o gradiente calculado a partir de mini-batches contém certo nível de ruído, o que pode provocar oscilações na função de perda. Se a taxa de aprendizado for muito baixa, a convergência será lenta; se for muito alta, o modelo poderá ultrapassar regiões de menor erro ou até divergir.

## SGD com Momentum

O Momentum é uma extensão do SGD que considera o histórico das atualizações realizadas anteriormente. Em vez de utilizar somente o gradiente atual, o método acumula uma espécie de velocidade na direção em que os gradientes têm sido consistentes.

As equações são:

$$
v_t = \mu v_{t-1} - \eta g_t
$$

$$
\theta_{t+1} = \theta_t + v_t
$$

Nessas equações, $v_t$ representa a velocidade acumulada e $\mu$ é o coeficiente de Momentum. Valores próximos de $0{,}9$ são frequentemente utilizados como ponto de partida.

A ideia pode ser comparada ao movimento de uma bola descendo uma superfície. Se os gradientes apontam repetidamente para uma mesma direção, as atualizações se acumulam e o modelo avança com maior velocidade. Quando os gradientes alternam de direção, o histórico reduz parte dessas oscilações.

Em redes densas, esse comportamento é especialmente útil em regiões da função de perda que apresentam vales estreitos. Nesses casos, o SGD puro pode movimentar-se de um lado para o outro, avançando lentamente em direção ao mínimo. O Momentum suaviza esse movimento e tende a acelerar a convergência.

## RMSProp

O RMSProp é um método adaptativo que modifica individualmente o tamanho das atualizações de cada parâmetro. Para isso, calcula uma média móvel dos gradientes elevados ao quadrado:

$$
s_t = \rho s_{t-1} + (1-\rho)g_t^2
$$

A atualização dos parâmetros é feita por:

$$
\theta_{t+1} = \theta_t -\frac{\eta}{\sqrt{s_t}+\varepsilon}g_t
$$

O termo $\varepsilon$ é um valor pequeno acrescentado para evitar uma divisão por zero.

A lógica do RMSProp é reduzir o tamanho do passo quando determinado parâmetro apresenta gradientes grandes ou frequentes. Em contrapartida, parâmetros associados a gradientes menores podem receber atualizações relativamente maiores. Dessa maneira, o algoritmo evita que alguns pesos dominem o processo de otimização.

Essa característica é relevante em redes densas porque diferentes camadas podem apresentar gradientes com escalas distintas. Enquanto uma camada pode receber atualizações muito grandes, outra pode apresentar gradientes pequenos. O RMSProp adapta os passos para lidar melhor com essa diferença, contribuindo para um treinamento mais estável [1].

## Adam

O Adam, abreviação de *Adaptive Moment Estimation*, combina o princípio do Momentum com a adaptação da taxa de aprendizado utilizada por métodos como o RMSProp. O algoritmo mantém duas médias móveis: uma dos gradientes e outra dos gradientes ao quadrado.

A primeira média é calculada por:

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

A segunda média é calculada por:

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

Como essas médias começam com valor zero, o Adam aplica uma correção de viés:

$$
\widehat{m}_t = \frac{m_t}{1-\beta_1^t}
$$

$$
\widehat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

A atualização final é dada por:

$$ 
\theta_{t+1}= \theta_t - \eta\frac{\widehat{m}_t}{\sqrt{\widehat{v}_t}+\varepsilon}
$$

O termo $m_t$ representa uma direção média dos gradientes e exerce função semelhante à do Momentum. Já $v_t$ acompanha a magnitude dos gradientes e permite adaptar o tamanho do passo para cada parâmetro.

Por combinar essas duas estratégias, o Adam geralmente apresenta uma redução rápida da perda no início do treinamento. Além disso, costuma ser uma opção prática quando se deseja iniciar um experimento sem realizar uma busca extensa pela taxa de aprendizado ideal. O artigo original do método destaca sua eficiência computacional, sua facilidade de implementação e sua adequação a problemas com grande quantidade de parâmetros [2].

## Comparação dos métodos

| Método | Princípio de funcionamento | Vantagens | Limitações |
|---|---|---|---|
| SGD puro | Utiliza apenas o gradiente atual | Simples, econômico e fácil de interpretar | Pode convergir lentamente e oscilar |
| SGD com Momentum | Acumula parte das atualizações anteriores | Reduz oscilações e acelera direções consistentes | Ainda utiliza uma taxa de aprendizado global |
| RMSProp | Ajusta o passo com base na média dos gradientes ao quadrado | Lida bem com gradientes de escalas diferentes | Pode depender de ajustes cuidadosos dos hiperparâmetros |
| Adam | Combina Momentum e adaptação individual dos parâmetros | Geralmente rápido, estável e prático | Nem sempre produz a melhor generalização |

A comparação deve considerar não apenas a velocidade de redução da perda de treinamento, mas também o desempenho no conjunto de validação. Um algoritmo pode diminuir rapidamente o erro dos dados utilizados no treinamento e, ainda assim, apresentar desempenho inferior em exemplos não vistos. Por esse motivo, é importante analisar simultaneamente a perda de treinamento, a perda de validação e as métricas específicas do problema.

## Aplicação em uma rede densa

Considere uma rede para classificação formada pelas camadas:

$$
784 \rightarrow 256 \rightarrow 128 \rightarrow 10
$$

Essa arquitetura pode receber uma entrada com 784 características, processá-la por duas camadas ocultas e produzir uma saída com 10 classes.

Com SGD puro, a rede poderá aprender de forma mais lenta e apresentar oscilações, principalmente se a taxa de aprendizado não estiver bem ajustada. O Momentum tende a acelerar o deslocamento em direções consistentes e reduzir essas oscilações. O RMSProp ajustará o tamanho do passo de acordo com o comportamento dos gradientes em cada parâmetro. O Adam combinará a suavização direcional do Momentum com a adaptação individual característica do RMSProp.

Em um experimento acadêmico, é importante manter constantes a arquitetura da rede, a divisão dos dados, o tamanho dos mini-batches, o número de épocas e as métricas avaliadas. Dessa forma, as diferenças observadas podem ser atribuídas principalmente ao algoritmo de otimização. O método mais adequado dependerá do objetivo: o SGD com Momentum pode ser interessante quando se busca uma boa generalização com ajuste cuidadoso, enquanto o Adam costuma ser uma escolha inicial eficiente para obter convergência rápida.

## Referências

[1] HINTON, Geoffrey. *Neural Networks for Machine Learning: Lecture 6a – Overview of Mini-Batch Gradient Descent*. University of Toronto. Disponível em: <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>. Acesso em: 30 set. 2026.

[2] KINGMA, Diederik P.; BA, Jimmy. *Adam: A Method for Stochastic Optimization*. International Conference on Learning Representations, 2015. Disponível em: <https://arxiv.org/abs/1412.6980>. Acesso em: 30 set. 2026.

[3] ZHANG, Aston; LIPTON, Zachary C.; LI, Mu; SMOLA, Alexander J. *Dive into Deep Learning: Optimization*. Disponível em: <https://pt.d2l.ai/chapter_optimization/index.html>. Acesso em: 30 set. 2026.

## Colaboradores

| |
|:---:|
| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/197432407?v=4" width="115"><br><sub>Beatriz Schuelter Tartare</sub>](https://github.com/beastartare) |
