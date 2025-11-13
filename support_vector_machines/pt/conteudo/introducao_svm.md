
Support Vector Machines (SVM) é uma técnica de aprendizado de máquina supervisionado muito utilizada na categorização de textos, na análise de imagens e na bioinformática. Foi fundamentada pela teoria de aprendizado estatístico (TAE) criada por Vapnik, que estabelece princípios para a obtenção de classificadores com boa generalização.

> [!NOTE]
> Generalização é definida como a capacidade do classificador de prever corretamente a classe de novos dados não apresentados previamente

O desempenho dos SVMs não se baseia apenas em resultados empíricos, mas também em um forte fundamento teórico como a máxima margem de separação, onde os SVMs buscam o hiperplano que maximiza a margem entre as classes. Isso reduz a chance de erro de generalização, ou seja, melhora o desempenho em dados novos.

> [!IMPORTANT]
> Por que maximizar a margem é bom? 🤔
> 
> A linha que está no meio do caminho, bem longe de todos os pontos de treinamento, é menos provável de classificar errado um ponto novo!

## Exemplo

Imagine que você está em um parque e há dois grupos de amigos brincando: um grupo de camisetas vermelhas e outro de camisetas azuis. Eles estão espalhados pelo gramado, mas há uma área onde os dois grupos estão mais próximos quase se misturando.
Agora, você quer esticar uma corda no chão para separar os dois grupos da melhor forma possível, sem passar por cima de ninguém. Mas não é só isso: você quer que a corda fique o mais distante possível dos amigos de cada grupo, para evitar confusões.

O que o SVM faz? 

- Ele procura a melhor posição para essa corda (o hiperplano) que separa os dois grupos.
- Os amigos mais próximos da corda são chamados de pontos de suporte — eles são os que “definem” onde a corda pode passar.
- A SVM tenta maximizar a distância entre a corda e esses amigos mais próximos, criando uma margem de segurança entre os grupos.

Se alguém mudar de lugar e ficar mais perto da corda, a posição dela pode mudar porque os pontos de suporte mudaram!

### Referência

Este projeto utiliza conceitos descritos no artigo "Uma Introdução às Support Vector Machines" (Ana Carolina Lorena e André C. P. L. F. de Carvalho).
https://www.researchgate.net/publication/36409205_Uma_Introducao_as_Support_Vector_Machines
