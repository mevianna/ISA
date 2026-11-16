# Evolução das Arquiteturas de Redes Neurais

## A Era Inicial e o Impacto das CNNs (LeNet a AlexNet)

Historicamente, o processamento de dados visuais por redes neurais artificiais enfrentava severos gargalos devido ao uso de Perceptrons Multicamadas (MLPs) totalmente conectados. Em imagens, cada neurônio de uma camada densa precisa se conectar a todos os valores da entrada. Por isso, o número de parâmetros pode se tornar muito grande à medida que aumentam a resolução da imagem e a quantidade de neurônios. As redes convolucionais reduziram esse problema ao explorar padrões locais e compartilhar os mesmos pesos em diferentes posições da imagem. A LeNet-5, apresentada por LeCun et al. (1998), foi um dos primeiros exemplos de grande impacto dessa abordagem, combinando convoluções e operações de redução espacial para reconhecer dígitos manuscritos.

Em 2012, a AlexNet (Krizhevsky et al., 2012) consolidou o *deep learning* ao vencer a competição ImageNet. A rede viabilizou o treinamento de arquiteturas mais profundas, com oito camadas, ao utilizar processamento em GPUs, a função de ativação ReLU e *dropout* para reduzir o *overfitting*.

## O Aprofundamento e a Padronização de Blocos (VGGNet e Inception)

Com o sucesso da AlexNet, a pesquisa focou em aumentar a profundidade das redes para extrair atributos mais complexos. A VGGNet (Simonyan e Zisserman, 2015) padronizou esse empilhamento ao substituir filtros grandes por sequências de filtros pequenos de $3 \times 3$, mantendo o campo receptivo com menos parâmetros e mais não linearidades.

Já a GoogLeNet/Inception (Szegedy et al., 2015) focou na eficiência computacional ao criar o módulo Inception. Esse bloco realiza convoluções de diferentes tamanhos em paralelo e utiliza convoluções $1 \times 1$ para reduzir a dimensionalidade antes de operações mais pesadas.

## O Problema da Degradação e a Barreira da Profundidade

À medida que os pesquisadores tentavam expandir as redes convolucionais, esperava-se que uma rede mais profunda apresentasse, no mínimo, o desempenho de uma rede mais rasa. Em teoria, as camadas adicionais poderiam simplesmente aprender um mapeamento de identidade, $H(x) = x$. Entretanto, experimentos mostraram que, depois de certa profundidade, o erro de treinamento podia aumentar.

Esse fenômeno é chamado de **degradação** e não é o mesmo que *overfitting*. No *overfitting*, o erro de treinamento é baixo, mas o desempenho em dados novos é ruim. Na degradação, a própria rede mais profunda apresenta erro de treinamento maior que o de uma rede mais rasa.

O desaparecimento ou a explosão dos gradientes pode dificultar o treinamento de redes profundas, mas não explica sozinho a degradação. No artigo original da ResNet, as redes já utilizavam técnicas como Batch Normalization. Mesmo assim, uma rede convencional de 56 camadas apresentou erro de treinamento maior que uma de 20 camadas. Isso indicava uma dificuldade de otimização: adicionar camadas tornava mais difícil encontrar uma boa solução, mesmo quando a rede mais profunda poderia, em princípio, reproduzir a solução da rede mais rasa.

## A Solução ResNet e o Aprendizado Residual (*Residual Learning*)

Para enfrentar a degradação, He et al. (2015) propuseram as Redes Residuais, ou ResNets. A inovação foi redefinir o objetivo de aprendizado: em vez de forçar um conjunto de camadas a aprender diretamente um mapeamento completo $H(x)$, o bloco aprende o resíduo:

$$
F(x) = H(x) - x
$$

Assim, o resultado desejado pode ser reescrito como:

$$
H(x) = F(x) + x
$$

O termo $F(x)$ é produzido pelas camadas do caminho principal. O termo $x$ segue por uma conexão de atalho (*shortcut* ou *skip connection*) e é somado ao resultado do caminho principal.

Em outras palavras, a entrada percorre dois caminhos: no caminho principal, passa pelas camadas que aprendem $F(x)$; no atalho, segue diretamente para a soma que produz $H(x)$.

Se a melhor transformação para determinado bloco estiver próxima da identidade, as camadas do caminho principal precisam aprender apenas um valor residual próximo de zero. Esse objetivo costuma ser mais fácil de otimizar do que reconstruir diretamente todo o mapeamento $H(x)$.

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

[1] LECUN, Yann et al. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, v. 86, n. 11, p. 2278–2324, 1998.

[2] KRIZHEVSKY, Alex; SUTSKEVER, Ilya; HINTON, Geoffrey E. ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, v. 25, 2012.

[3] SIMONYAN, Karen; ZISSERMAN, Andrew. Very deep convolutional networks for large-scale image recognition. *International Conference on Learning Representations*, 2015.

[4] SZEGEDY, Christian et al. Going deeper with convolutions. In: *IEEE Conference on Computer Vision and Pattern Recognition*. 2015. p. 1–9.

[5] HE, Kaiming et al. Deep residual learning for image recognition. In: *IEEE Conference on Computer Vision and Pattern Recognition*. 2016. p. 770–778. Disponível em: <https://arxiv.org/abs/1512.03385>. Acesso em: 15 nov. 2026.

[6] ZHANG, Aston et al. *Dive into Deep Learning*. Cambridge: Cambridge University Press, 2024. Seção 8.6: Residual Networks (ResNet) and ResNeXt. Disponível em: <https://d2l.ai/chapter_convolutional-modern/resnet.html>. Acesso em: 15 nov. 2026.

[7] CHOLLET, François. *Deep Learning with Python*. 2. ed. Shelter Island: Manning Publications, 2021. Cap. 9, seção 9.3.2: Residual connections.

[8] ELGENDY, Mohamed. *Deep Learning for Vision Systems*. Shelter Island: Manning Publications, 2020. Cap. 5, seção 5.6: ResNet, p. 230–239.
