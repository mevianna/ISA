# Portuguese Version
# English Version
## Naive Bayes Explained
Naive Bayes is a family of probabilistic algorithms based on Baye's theorem and its commonly used for classification tasks. Before understanting about naive bayes algorithms let's first jump into [Baye's Theorem](baye's_theorem_explained.md).

Now that we understand the basis of bayes rule let's understand a litle more of naive-bayes.

Naive Bayes uses probability to predict which category a data point belongs to, considering that all features, this is the data's characterisctics or atributtes, are independent and this simplifies the computation significantly.

> [!IMPORTANT]
> - Naive Bayes Classifier can predict at a faster speed than others classification algotithms because it's a simple probabilistic classifier and it doesn't have many parameters to build the ML models.
> - It is a probabilistic classifier because it assumes that one feature in the model is independent of existence of another feature. That means that each feature contributes to the predictions and has no relation between each other.

### Why the name *Naive Bayes*? 🤔
The term "Naive" comes from the fact that the model assumes that the presence of one feature does not affact other features, that the characteristic or attribute of the data are independent. And the "Bayes" part comes from to its basis in Bayes' Theorem.

### Types of Naive Bayes Model 📩
There are 3 types for classification in this model
1. Gaussian Naive Bayes
2. Multinomial Naive Bayes
3. Bernoulli Naive Bayes

The **first one** is speciaaly used for continuous data where features follow a Gaussian (normal) distribuition.  
The **second one**, also known as MNB, is well-suited for text classification tasks where term frequencies such as words counts in text are important.  
The **third one** deals with binary features like if a word appears or not in a document and it is often used in document classification tasks.

### Advantages of Naive Bayes Classifier 👍
- Simple to use and fast to compute.
- Works well even with many different characteristics.
- Good results even with a small amount of training information.
- Excellent with data that falls into categories (like "spam" or "not spam").
- For number-based data is assumed to come from normal distributions.

### Disadvantages of Naive Bayes Classifier 👎
- Assumes that all characteristics are separate from each other, which isn't always true in real life.  
- Can be affected by information that isn't important.  
- Might struggle with new information it hasn't seen before, which can lead to bad predictions.

### Applications of Naive Bayes Classifier 💡
- Spam Email Filtering: Sorts emails into "spam" or "not spam."
- Text Classification: Used for understanding feelings in text, sorting documents, and finding topics.
- Medical Diagnosis: Helps guess the chance of someone having a disease based on their symptoms.
- Credit Scoring: Checks how likely someone is to pay back a loan.
- Weather Prediction: Forecasts weather based on different factors.

### References
- <[Naive Bayes Classifiers](https://www.geeksforgeeks.org/naive-bayes-classifiers/)>. Acess 29 may 2025 



