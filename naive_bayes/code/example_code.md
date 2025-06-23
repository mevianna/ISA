# Exemplo: Classificação de e-mails com Naive Bayes

O principal objetivo deste algoritmo é demonstrar como o método Naive Bayes pode ser muito eficaz em tarefas de classificação, como a detecção de spam. A seguir, apresentamos um passo a passo didático e de fácil compreensão, para que você possa reproduzir a implementação com outros conjuntos de dados e, assim, praticar e se divertir! :smile:




<img src="https://media1.tenor.com/m/3AQDvhSiPpMAAAAC/dog-hacker.gif" width="300" />


## Código

### Importar bibliotecas
``` python
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
```
- pandas as pd → para ler o CSV e manipular o DataFrame.

- MultinomialNB → classificador Naive Bayes.

- train_test_split → divisão entre treino e teste.

- TfidfVectorizer → (importado dentro do código, mas usado corretamente).

- SMOTE → para balancear os dados.
  
- TfidfVectorizer → usada para transformar texto em números.

- Counter → é uma estrutura de dados que serve para contar a frequência de elementos em um iterável.

- confusion_matrix, seaborn, matplotlib.pyplot → para visualizar a matriz de confusão.

### Carregar e preparar os dados
``` python
ds = pd.read_csv("/content/spam.csv")
ds["spam"] = ds["Category"].apply(lambda x: 1 if x == "spam" else 0)
y = ds.iloc[:, 2].values
X = ds["Message"]
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)
X_treino, X_teste, y_treino, y_teste = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
```
- Carrega o dataset CSV → le o arquivo CSV chamado spam.csv e salva o conteúdo no DataFrame ds usando a biblioteca pandas.
  
- Cria uma nova coluna "spam" com rótulos binários (0 e 1).
  - Acessa a coluna Category, que contém rótulos de texto como "spam" ou "ham" (não-spam)
  - Usa apply() com uma função lambda para converter esses textos em números: "spam" → 1, qualquer outro valor (como "ham") → 0
  - Cria uma nova coluna chamada "spam" no DataFrame com esses valores
    
- Separa os rótulos (y).
  
- Separa as mensagens (X).

- Vetorização com TfidfVectorizer → transforma os textos da coluna Message em vetores numéricos esparsos com base na frequência e relevância de cada palavra (TF-IDF = Term Frequency – Inverse Document Frequency).

- Divide os dados em 80% para treino e 20% para teste → stratify=y garante que a proporção de classes (spam/ham) seja mantida igual nas duas partes.

> [!Note]
>Por que usar TF-IDF?
>  - Dá peso maior às palavras que realmente têm importância.
>  - Ignora palavras comuns (como "e", "a", "de", etc.).
>  - É rápido e eficiente.
>  - Funciona muito bem com modelos lineares como Naive Bayes.

### Oversampling
``` python
print("Antes do oversampling:", Counter(y_treino))
```
``` python
smote = SMOTE(random_state=42)
X_treino_smote, y_treino_smote = smote.fit_resample(X_treino, y_treino)
```
``` python
print("Após o oversampling:", Counter(y_treino_smote))
```

### Treinar, testar e visualizar os resultados
``` python
modelo = MultinomialNB()
modelo.fit(X_treino_smote, y_treino_smote)
previsoes = modelo.predict(X_teste)
previsoes
```
``` python
y_teste
```
``` python
modelo.score(X_teste, y_teste)
```
``` python
cm = confusion_matrix(y_teste, previsoes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
```
