# Pandas e Manipulação de Dados

**Grupo 2 · Nivelamento ISA - LIA** · Albie e Mayara

## Sobre este conteúdo

O **Pandas** é a biblioteca do Python mais usada para trabalhar com dados em tabela. Neste material explicamos, com um exemplo real, como criar, carregar, explorar, selecionar e manipular tabelas de dados, que é o primeiro passo de praticamente todo projeto de Machine Learning.

## Conteúdo

| # | Tema | Arquivo |
|---|---|---|
| 1 | Introdução ao Pandas e Estruturas de Dados |  |
| 2 | Importação e Exploração de Dados |  |
| 3 | Seleção e Filtragem de Dados | [3.selecao_filtragem_dados.md](3.selecao_filtragem_dados.md) |
| 4 | Manipulação de DataFrames | [4.manipulacao_dataframes.md](4.manipulacao_dataframes.md) |

## Notebook

Todos os exemplos estão no notebook [02_pandas_manipulacao_dados.ipynb](02_pandas_manipulacao_dados.ipynb), que pode ser aberto no Google Colab.

## Dataset usado nos exemplos

**Aluguéis no Brasil:** cerca de 10 mil imóveis para alugar em São Paulo, Rio de Janeiro, Belo Horizonte, Porto Alegre e Campinas, com área, quartos, banheiros, vagas, andar, se aceita animais, se é mobiliado e os valores (aluguel, condomínio, IPTU, seguro e total).

- Fonte: OpenML, dataset *Brazilian_houses* (ID 42688)
- Origem: 2ª versão do dataset "Brazilian houses to rent", publicado originalmente no Kaggle

**Pergunta que guia os exemplos:** qual cidade tem o aluguel mais caro, e o que faz um imóvel ser mais caro?

## Relação com os outros temas do nivelamento

- **Grupo 1 (NumPy):** o Pandas é construído em cima do NumPy; cada coluna de um DataFrame guarda os dados num array.
- **Grupo 3 (Limpeza):** depois de saber manipular tabelas, o próximo passo é tratar valores ausentes, duplicados e outliers.
- **Grupo 4 (Estatística e EDA):** usa o Pandas para calcular estatísticas e fazer gráficos.

## Referências

- Documentação oficial do Pandas: https://pandas.pydata.org/docs/
- Guia "10 minutes to pandas": https://pandas.pydata.org/docs/user_guide/10min.html

