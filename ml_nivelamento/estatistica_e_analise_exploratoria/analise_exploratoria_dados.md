 - [Estatística e Análise Exploratória](#h1)
    - [Análise Exploratória de Dados (EDA)](#h2)
        - [O que é EDA?](#h3_0)  
        - [Objetivos da análise exploratória](#h3_1)    
        - [Identificação de padrões](#h3_2)    
        - [Identificação de relações entre variáveis](#h3_3)    
        - [Identificação de possíveis problemas nos dados](#h3_4) 


# <a id='h1'></a> Estatística e Análise Exploratória

## <a id='h2'></a> Análise Exploratória de Dados (EDA)

### <a id='h3_0'></a> O que é EDA?

A Análise Exploratória de Dados (Exploratory Data Analysis - EDA) é a filosofia e a abordagem analítica dedicada a investigar, resumir e visualizar as características fundamentais de um conjunto de dados antes da aplicação de modelos estatísticos formais, testes de hipóteses ou algoritmos de aprendizado de máquina.


O termo e a metodologia foram formalmente consolidados pelo estatístico americano John W. Tukey no clássico livro "Exploratory Data Analysis" (1977). Tukey defendia que, em vez de iniciar uma análise tentando confirmar hipóteses pré-concebidas (Confirmatory Data Analysis - CDA), o analista deve atuar como um detetive: explorar os dados de forma flexível, permitindo que os próprios dados revelem suas estruturas e inconsistências.


Na literatura estatística moderna, como em Montgomery & Runger (Estatística Aplicada e Probabilidade para Engenheiros), a EDA é apresentada como uma etapa essencial da Estatística Descritiva Aplicada, fornecendo as bases empíricas para validar todas as premissas matemáticas que sustentam a inferência estatística.


Os Três Pilares Teóricos da EDA:

- Diagnóstico de Qualidade dos Dados
- Análise Univariada
- Análise Bivariada e Multivariada


### <a id='h3_1'></a> Objetivos da análise exploratória

- Maximizar a Compreensão da Estrutura dos Dados
- Detectar Anomalias, Erros e Outliers
- Extrair Relações, Padrões e Estruturas Ocultas


### <a id='h3_2'></a> Identificação de padrões

A identificação de padrões na Análise Exploratória de Dados consiste em reconhecer formas relativas à distribuição, tendência, agrupamentos e comportamentos estruturais nos dados.


1. Padrões de Distribuição e Forma da Curva

Consiste em analisar como os dados estão distribuídos e qual é o formato assumido pela sua curva de frequência. Para isso, podem ser utilizadas medidas como média, mediana e moda, além de medidas de dispersão, como variância, desvio padrão e intervalo interquartil.

A comparação entre essas medidas permite perceber, por exemplo, se os dados estão concentrados em torno de determinado valor, se apresentam assimetria ou se possuem grande variabilidade. A análise pode ser complementada por histogramas e boxplots, que permitem visualizar a concentração dos valores, a amplitude da distribuição e a possível presença de valores discrepantes.

Dessa maneira, a distribuição dos dados fornece uma visão inicial do comportamento da variável e auxilia na identificação de estruturas que poderiam não ser percebidas apenas pelas medidas numéricas.

2. Padrões de Tendência Central e Variabilidade

Permitem compreender em torno de quais valores os dados se concentram e o quanto eles se afastam dessa região central.

A média, mediana e moda fornecem diferentes perspectivas sobre a tendência central, enquanto a amplitude, variância, desvio padrão e intervalo interquartil ajudam a avaliar a dispersão dos dados.

Observar simultaneamente tendência central e variabilidade é importante para evitar uma interpretação baseada apenas em um único valor representativo e para compreender melhor a estrutura do conjunto de dados.

3. Padrões de Agrupamento e Lacunas

Relacionados à forma como os valores se concentram ou se distribuem ao longo do conjunto de dados.

Um agrupamento ocorre quando uma quantidade significativa de observações se concentra em determinada faixa de valores, podendo indicar a existência de diferentes grupos ou comportamentos dentro da amostra.

Já as lacunas correspondem a regiões da distribuição nas quais existem poucos ou nenhum dado, podendo representar uma característica real do fenômeno ou indicar alguma particularidade na coleta dos dados.

Histogramas, boxplots e gráficos de dispersão são ferramentas úteis para visualizar esses padrões.

A identificação desses agrupamentos pode ser particularmente importante quando há diferenças entre subgrupos, enquanto as lacunas devem ser analisadas com atenção para verificar se representam uma característica natural dos dados ou algum problema na coleta. Assim, a identificação de agrupamentos e lacunas contribui para compreender estruturas que poderiam ficar ocultas quando são observadas apenas medidas como média e mediana

4. Padrões Temporais e Sequenciais

São identificados quando os dados possuem uma ordem relacionada ao tempo ou à sequência de ocorrência das observações.

Nesse caso, além de analisar média, mediana e dispersão, é necessário observar como os valores se modificam ao longo do tempo. Podem ser identificadas tendências de crescimento ou redução, oscilações, períodos de maior ou menor variabilidade e comportamentos recorrentes.

A representação gráfica é especialmente importante nesse tipo de análise, pois permite observar visualmente mudanças que poderiam não ser evidentes por meio das estatísticas descritivas isoladamente.

Em uma análise exploratória, reconhecer esses padrões ajuda a compreender a dinâmica dos dados antes da aplicação de modelos estatísticos mais formais.


### <a id='h3_3'></a> Identificação de relações entre variáveis

A investigação das relações entre variáveis busca mapear como a variação de uma determinada métrica acompanha a alteração de outra.

1. Direção da Variação Conjunta

Busca verificar como duas variáveis se comportam simultaneamente. Essa relação pode ser investigada por meio de tabelas, gráficos de dispersão e medidas estatísticas, especialmente a correlação e a covariância.

O gráfico de dispersão permite observar diretamente a disposição dos pares de valores e identificar se existe algum padrão de variação conjunta. É importante destacar que essa análise descreve a associação entre as variáveis, não permitindo, por si só, concluir que uma variável provoca alterações na outra.


2. Coeficiente de Correlação Amostral

É uma medida utilizada para quantificar a intensidade e a direção da associação entre duas variáveis, especialmente quando se investiga uma relação linear.

Entretanto, a interpretação da correlação deve considerar o contexto e a distribuição dos dados, pois um coeficiente próximo de zero não significa necessariamente que não exista nenhuma relação entre as variáveis, já que pode existir uma relação não linear.

O coeficiente deve, portanto, ser analisado em conjunto com o gráfico de dispersão e com outras características dos dados.

3. Relações Lineares (Positivas e Negativas)

Ocorrem quando a variação entre duas variáveis pode ser aproximadamente representada por uma tendência em linha reta.

Em uma relação linear positiva, valores maiores de uma variável tendem a estar associados a valores maiores da outra, enquanto em uma relação linear negativa, valores maiores de uma variável tendem a estar associados a valores menores da outra.

O gráfico de dispersão é uma ferramenta importante para identificar visualmente esse comportamento, enquanto o coeficiente de correlação fornece uma medida numérica da direção e intensidade da associação linear. É importante, entretanto, observar a dispersão dos pontos, pois uma correlação elevada pode ser influenciada por valores extremos ou por características específicas da amostra.

4. Relações Não-Lineares / Curvilíneas

Nem todas as relações entre variáveis podem ser representadas adequadamente por uma reta. Em algumas situações, uma variável pode aumentar inicialmente com o crescimento de outra e, posteriormente, diminuir, formando uma relação curvilínea ou não linear.

Nesses casos, o coeficiente de correlação linear pode apresentar um valor baixo mesmo existindo uma relação evidente entre as variáveis. Por isso, a análise exploratória deve utilizar recursos de visualização, principalmente o gráfico de dispersão, para identificar formatos como curvas, parábolas ou outros padrões.

5. Ausência de Associação

A ausência de associação ocorre quando não é identificado um padrão sistemático que relacione as variações de duas variáveis.

6. Variáveis de Confusão

As variáveis de confusão devem ser consideradas quando uma terceira variável pode estar relacionada simultaneamente às duas variáveis que estão sendo analisadas, criando ou alterando a associação observada entre elas.

Dessa forma, uma correlação identificada entre duas variáveis não deve ser interpretada automaticamente como uma relação direta entre elas.

A análise exploratória pode ajudar a levantar esse tipo de possibilidade por meio da comparação entre diferentes grupos, da análise de gráficos e da investigação de outras variáveis disponíveis no conjunto de dados.

A identificação de possíveis variáveis de confusão é importante porque permite interpretar a associação encontrada com maior cautela e evita atribuir diretamente a uma variável um comportamento que pode estar relacionado a outro fator.

7. Inversão da Causalidade

A inversão da causalidade ocorre quando a relação observada entre duas variáveis é interpretada em uma direção que pode não corresponder ao mecanismo real do fenômeno.

Por isso, é necessário distinguir cuidadosamente correlação, associação e causalidade. Essa preocupação é particularmente importante porque a simples existência de uma correlação positiva ou negativa não permite determinar qual variável influencia a outra.

8. Comprovação de Causalidade

A identificação de uma relação entre duas variáveis durante a EDA não é suficiente para comprovar que uma delas causa alterações na outra.

A análise exploratória tem como função principal investigar, resumir e revelar padrões e relações presentes nos dados, servindo como uma etapa anterior à aplicação de métodos estatísticos formais.

Assim, uma correlação encontrada por meio de medidas estatísticas ou de um gráfico de dispersão deve ser interpretada como uma evidência de associação, e não automaticamente como evidência de causalidade.

A comprovação de uma relação causal exige análises adicionais e um desenho de estudo adequado ao problema investigado, considerando possíveis variáveis de confusão e a direção da relação.

Portanto, dentro da EDA, a causalidade deve ser tratada como uma questão que precisa de investigação adicional, e não como uma conclusão obtida simplesmente pela observação de correlação.

### <a id='h3_4'></a> Identificação de possíveis problemas nos dados

A etapa final do ciclo da Análise Exploratória de Dados (EDA) foca no diagnóstico e na identificação de anomalias que possam comprometer a integridade dos dados e a validade de futuras análises estatísticas ou modelos preditivos.


1. Outliers

Os outliers são valores que se distanciam significativamente do comportamento geral de um conjunto de dados e podem surgir devido a erros de medição ou digitação, como instrumentos mal calibrados ou registros incorretos. Entretanto, nem todo outlier representa um erro, pois ele também pode estar relacionado a eventos raros ou à variabilidade natural do fenômeno analisado. Para identificar possíveis valores discrepantes, pode-se utilizar a Regra do Intervalo Interquartil (IQR).

2. Dados Faltantes

Os dados faltantes correspondem à ausência de informações em um conjunto de dados, o que pode reduzir o poder estatístico da amostra e introduzir viés nos resultados.

Esses dados podem ser classificados em três mecanismos:
- Missing Completely at Random, quando a ausência não depende de nenhuma variável, observada ou não;
- Missing at Random, quando a ausência está relacionada a outras variáveis observadas no dataset, mas não ao próprio valor faltante;
- Missing Not at Random, quando a ausência depende diretamente do valor que deveria estar presente, como no caso de indivíduos com renda muito elevada que optam por não declarar seus ganhos.

Algumas estratégias podem ser utilizadas para tratar esses valores, como a remoção de registros ou variáveis, especialmente quando a quantidade de dados faltantes é pequena e o mecanismo é MCAR, e a imputação estatística, que consiste em substituir os valores ausentes por medidas como média ou mediana ou por estimativas obtidas por meio de modelos preditivos.


3. Inconsistência de Formato e Desbalanceamento

As inconsistências de formato e os erros estruturais podem comprometer a qualidade dos dados e a confiabilidade das análises, ocorrendo, por exemplo, quando variáveis numéricas são armazenadas como texto, datas apresentam formatos diferentes ou existem valores duplicados no conjunto de dados. 

Também é importante verificar a presença de violações dos limites de domínio, como percentuais fora do intervalo de 0% a 100% ou valores fisicamente impossíveis, como eficiências térmicas negativas.

Além disso, o desbalanceamento de classes ocorre quando uma categoria representa uma parcela muito maior dos registros do que as demais. A identificação dessas inconsistências e do desbalanceamento durante a Análise Exploratória de Dados (EDA) é fundamental para garantir a qualidade da base e evitar que modelos preditivos posteriores sejam excessivamente influenciados pela classe majoritária.