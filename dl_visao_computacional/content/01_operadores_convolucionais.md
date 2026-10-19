# Operadores Convolucionais

## O que são operadores convolucionais?
Os operadores convolucionais são utilizados em redes neurais convolucionais (CNNs) para processar imagens e identificar características relevantes em seus pixels. Para isso, filtros percorrem pequenas regiões da imagem, realizando operações matemáticas que permitem destacar padrões como bordas, texturas e formas. O resultado desse processo depende da maneira como os filtros são aplicados e dos parâmetros utilizados na operação.

> **Nota:** Redes Neurais Convolucionais (CNNs) são um tipo de rede neural utilizado principalmente no processamento e na análise de imagens. Sua principal característica é o uso de operações convolucionais para identificar padrões visuais, como bordas, texturas e formas, que podem ser utilizados em tarefas como classificação de imagens e detecção de objetos.

## Kernels (filtros convolucionais)
Os kernels são pequenas matrizes numéricas utilizadas na operação de convolução para analisar regiões de uma imagem. Durante o processamento, os valores do kernel são multiplicados pelos valores dos pixels correspondentes, e os resultados dessas multiplicações são somados, produzindo um único valor para cada posição analisada. Esse processo é repetido ao longo da imagem, permitindo identificar diferentes características visuais.

Em uma CNN, os valores dos kernels geralmente são ajustados durante o treinamento da rede, de modo que os filtros aprendam a reconhecer padrões relevantes para a tarefa proposta.

## Referências
[1] O que é uma rede neural convolucional? Disponível em: <https://cloud.google.com/discover/what-are-convolutional-neural-networks?hl=pt-BR>. Acesso em: 19 out. 2026.


