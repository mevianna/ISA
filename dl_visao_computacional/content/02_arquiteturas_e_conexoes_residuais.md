# Evolução das Arquiteturas de Redes Neurais

## A Era Inicial e o Impacto das CNNs (LeNet a AlexNet)

Historicamente, o processamento de dados visuais por redes neurais artificiais enfrentava severos gargalos devido ao uso de Perceptrons Multicamadas (MLPs) totalmente conectados. Em dados de alta dimensionalidade, como imagens digitais, o número de parâmetros exigidos por arquiteturas densas é proporcional ao produto entre o tamanho da entrada e o número de neurônios, podendo tornar o treinamento computacionalmente inviável e altamente suscetível ao overfitting. A transição para a extração hierárquica de atributos teve um de seus primeiros grandes marcos com a introdução da LeNet-5 por (LeCun et al. 1998). A LeNet-5 consolidou o uso de operadores de convolução e subsampling (pooling), permitindo que o modelo aprendesse invariâncias espaciais locais e reduzisse drasticamente o número de parâmetros livres ao compartilhar pesos no espaço vetorial.

Em 2012, a AlexNet (Krizhevsky et al., 2012) consolidou o Deep Learning ao vencer a competição ImageNet. A rede viabilizou o treinamento de arquiteturas mais profundas (8 camadas) ao usar processamento em GPUs, a função de ativação ReLU que amenizou o problema do vanishing gradient e o Dropout para evitar overfitting.

## O Aprofundamento e a Padronização de Blocos (VGGNet e Inception)

Com o sucesso da AlexNet, a pesquisa focou em aumentar a profundidade das redes para extrair atributos mais complexos. A VGGNet (Simonyan & Zisserman, 2014) padronizou esse empilhamento ao substituir filtros grandes por sequências de filtros pequenos de 3 x 3, o que manteve o campo receptivo, reduziu parâmetros e adicionou mais não-linearidades. 

Já a GoogLeNet/Inception (Szegedy et al., 2015) focou na eficiência computacional ao criar o módulo Inception. Esse bloco realiza convoluções de diferentes tamanhos (1 x 1,3 x 3,5 x 5) em paralelo, aplicando convoluções 1 x 1 (bottlenecks) para reduzir a dimensionalidade antes de operações mais pesadas.

## O Problema da Degradação e a Barreira da Profundidade

À medida que os pesquisadores tentavam expandir as redes convolucionais para além de 20 ou 30 camadas, a busca por maior profundidade encontrou uma barreira teórica e prática fundamental. Esperava-se que redes mais profundas apresentassem, no mínimo, um desempenho idêntico ao de suas contrapartes mais rasas, visto que as camadas adicionais poderiam teoricamente aprender mapeamentos de identidade (H(x) = x). Contudo, experimentos demonstraram empiricamente o oposto: a partir de um certo limite de profundidade, a acurácia do modelo saturava e, em seguida, degradava rapidamente.

É crucial destacar que esse fenômeno de degradação difere fundamentalmente do overfitting. Enquanto o overfitting se caracteriza por um baixo erro no conjunto de treinamento acompanhado de uma baixa generalização no conjunto de teste, o problema da degradação manifesta-se com o aumento do próprio erro de treinamento. O aumento da profundidade em arquiteturas estritamente sequenciais torna a otimização mais difícil. O desaparecimento do gradiente (vanishing gradient) também pode prejudicar a atualização das camadas iniciais, mas não explica sozinho o problema da degradação observado nas redes profundas.

## A Solução ResNet e o Aprendizado Residual (Residual Learning)

Para superar a degradação, (He et al. 2015) propuseram as ResNets. A inovação foi redefinir o objetivo de aprendizado: em vez de forçar as camadas a aprender um mapeamento completo H(x), a rede aprende o resíduo F(x) := H(x) - x.

Essa ideia é aplicada usando atalhos de salto (skip connections), que somam a entrada diretamente à saída do bloco (H(x) = F(x) + x). Esses atalhos oferecem um caminho direto para a propagação do gradiente até as camadas iniciais durante o backpropagation, facilitando o treinamento de redes com centenas de camadas.

## Como a Conexão Residual Ajuda o Gradiente

Considere a saída simplificada de um bloco:

$$
y = F(x) + x
$$

Durante o *backpropagation*, a derivada em relação a $x$ contém dois caminhos:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\left(
\frac{\partial F(x)}{\partial x} + 1
\right)
$$

O termo $\frac{\partial F(x)}{\partial x}$ vem do caminho com as camadas convolucionais. O termo $1$ vem da conexão identidade. Dessa forma, o gradiente dispõe de um caminho direto entre blocos, o que facilita sua propagação pelas partes iniciais da rede.

Isso não significa que o gradiente nunca desaparecerá nem que qualquer rede poderá ser aprofundada sem limites. A conexão residual apenas torna a otimização de redes profundas mais estável e, em geral, mais fácil.

## O Bloco Residual Básico

Na ResNet-18 e na ResNet-34, o bloco básico possui normalmente:

1. uma convolução $3 \times 3$, seguida de Batch Normalization e ReLU;
2. outra convolução $3 \times 3$, seguida de Batch Normalization;
3. a soma entre o caminho principal e o atalho;
4. uma nova aplicação da ReLU.

Para que a soma entre o caminho principal e o atalho seja possível, os dois tensores precisam ter as mesmas dimensões.

## Quando as Dimensões são Diferentes

Ao avançar de um estágio da ResNet para outro, é comum reduzir a altura e a largura dos mapas de características e aumentar o número de canais. Nesse caso, $F(x)$ e $x$ deixam de ter o mesmo formato e não podem ser somados diretamente.

Uma solução é usar, no atalho, uma convolução $1 \times 1$ com o *stride* necessário:

$$
y = F(x) + W_s x
$$

Aqui, $W_s$ representa a projeção que ajusta o formato de $x$. Por exemplo, uma convolução $1 \times 1$ com *stride* igual a 2 pode reduzir as dimensões espaciais, enquanto seus filtros ajustam a quantidade de canais. Quando os formatos já são iguais, o atalho identidade é preferível porque não adiciona parâmetros.

## Bloco Básico e Bloco *Bottleneck*

As principais versões da ResNet usam dois tipos de bloco:

| Versões | Bloco utilizado | Estrutura do caminho principal |
|---|---|---|
| ResNet-18 e ResNet-34 | bloco básico | duas convoluções $3 \times 3$ |
| ResNet-50, ResNet-101 e ResNet-152 | *bottleneck* | convoluções $1 \times 1$, $3 \times 3$ e $1 \times 1$ |

No *bottleneck*, a primeira convolução $1 \times 1$ reduz o número de canais processados pela convolução $3 \times 3$; a última restaura a quantidade de canais da saída. Essa organização permite construir redes mais profundas com um custo computacional mais controlado.

Uma ResNet mais profunda não é automaticamente a melhor escolha para qualquer problema. Modelos maiores exigem mais memória e processamento. A ResNet-18, por exemplo, costuma ser uma alternativa simples para experimentos e para *transfer learning*.

## Em Resumo

- redes simplesmente mais profundas podem ser mais difíceis de otimizar;
- um bloco residual aprende $F(x)$ e soma a entrada $x$ ao resultado;
- a conexão identidade cria um caminho direto para a informação e para o gradiente;
- uma projeção $1 \times 1$ ajusta o atalho quando os formatos são diferentes;
- ResNet-18/34 usam blocos básicos, enquanto ResNet-50/101/152 usam blocos *bottleneck*.

## Referências

[1] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

[2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012).

[3] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

[4] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[5] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[6] ZHANG, Aston et al. *Dive into Deep Learning*. Cambridge: Cambridge University Press, 2024. Seção 8.6: Residual Networks (ResNet) and ResNeXt. Disponível em: <https://d2l.ai/chapter_convolutional-modern/resnet.html>. Acesso em: 15 nov. 2026.

[7] CHOLLET, François. *Deep Learning with Python*. 2. ed. Shelter Island: Manning Publications, 2021. Cap. 9, seção 9.3.2: Residual connections.

[8] ELGENDY, Mohamed. *Deep Learning for Vision Systems*. Shelter Island: Manning Publications, 2020. Cap. 5, seção 5.6: ResNet, p. 230–239.
