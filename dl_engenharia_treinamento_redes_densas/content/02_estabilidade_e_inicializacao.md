# Técnicas de inicialização de parâmetros (Xavier/He)

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


## Colaboradores

| |
|:---:|
| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/112569754?v=4" width="115"><br><sub>Alice Motin Bastos </sub>](https://github.com/AliceMotin) |
