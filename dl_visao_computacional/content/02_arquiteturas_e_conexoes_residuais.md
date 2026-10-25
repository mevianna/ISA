# Evolução das Arquiteturas de Redes Neurais

## A Era Inicial e o Impacto das CNNs (LeNet a AlexNet)

Historicamente, o processamento de dados visuais por redes neurais artificiais enfrentava severos gargalos devido ao uso de Perceptrons Multicamadas (MLPs) totalmente conectados. Em dados de alta dimensionalidade, como imagens digitais, o número de parâmetros exigidos por arquiteturas densas cresce exponencialmente, tornando o treinamento computacionalmente inviável e altamente suscetível ao overfitting. A transição para a extração hierárquica de atributos teve seu primeiro grande marco com a introdução da LeNet-5 por (LeCun et al. 1998). A LeNet-5 introduziu formalmente os operadores de convolução e subsampling (pooling), permitindo que o modelo aprendesse invariâncias espaciais locais e reduzisse drasticamente o número de parâmetros livres ao compartilhar pesos no espaço vetorial.

Em 2012, a AlexNet (Krizhevsky et al., 2012) consolidou o Deep Learning ao vencer a competição ImageNet. A rede viabilizou o treinamento de arquiteturas mais profundas (8 camadas) ao usar processamento em GPUs, a função de ativação ReLU que amenizou o problema do vanishing gradient e o Dropout para evitar overfitting.

## O Aprofundamento e a Padronização de Blocos (VGGNet e Inception)

Com o sucesso da AlexNet, a pesquisa focou em aumentar a profundidade das redes para extrair atributos mais complexos. A VGGNet (Simonyan & Zisserman, 2014) padronizou esse empilhamento ao substituir filtros grandes por sequências de filtros pequenos de 3 x 3, o que manteve o campo receptivo, reduziu parâmetros e adicionou mais não-linearidades. 

Já a GoogLeNet/Inception (Szegedy et al., 2015) focou na eficiência computacional ao criar o módulo Inception. Esse bloco realiza convoluções de diferentes tamanhos (1 x 1,3 x 3,5 x 5) em paralelo, aplicando convoluções 1 x 1 (bottlenecks) para reduzir a dimensionalidade antes de operações mais pesadas.

## O Problema da Degradação e a Barreira da Profundidade

À medida que os pesquisadores tentavam expandir as redes convolucionais para além de 20 ou 30 camadas, a busca por maior profundidade encontrou uma barreira teórica e prática fundamental. Esperava-se que redes mais profundas apresentassem, no mínimo, um desempenho idêntico ao de suas contrapartes mais rasas, visto que as camadas adicionais poderiam teoricamente aprender mapeamentos de identidade (H(x) = x). Contudo, experimentos demonstraram empiricamente o oposto: a partir de um certo limite de profundidade, a acurácia do modelo saturava e, em seguida, degradava rapidamente.

É crucial destacar que esse fenômeno de degradação difere fundamentalmente do overfitting. Enquanto o overfitting se caracteriza por um baixo erro no conjunto de treinamento acompanhado de uma baixa generalização no conjunto de teste, o problema da degradação manifesta-se com o aumento do próprio erro de treinamento. O aumento da profundidade em arquiteturas estritamente sequenciais torna a superfície de otimização excessivamente complexa, dificultando a propagação contínua do sinal de erro através da retropropagação do gradiente (backpropagation). O gradiente diminui exponencialmente à medida que se aproxima das camadas iniciais (vanishing gradient), impedindo a atualização eficaz dos pesos e criando um limite prático para a profundidade viável dos modelos tradicionais.

## A Solução ResNet e o Aprendizado Residual (Residual Learning)

Para superar a degradação, (He et al. 2015) criaram as ResNets. A inovação foi redefinir o objetivo de aprendizado: em vez de forçar as camadas a aprender um mapeamento completo H(x), a rede aprende o resíduo F(x) := H(x) - x.

Essa ideia é aplicada usando atalhos de salto (skip connections), que somam a entrada diretamente à saída do bloco (H(x) = F(x) + x). Esses atalhos permitem que o gradiente flua sem obstáculos até as camadas iniciais durante o backpropagation, viabilizando o treinamento de redes com centenas de camadas.

## Referências
[1] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

[2] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012).

[3] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

[4] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

[5] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
