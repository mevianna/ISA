# English Version
# Baye's Theorem
Bayes theorem (also known as the Bayes Rule or Bayes Law) is used to determine the conditional probability of event A when event B has already occurred, it's a formula used to calculate the probability of an outcome given a known previus outcome that occured in similar circumstances.
> KEY TAKEAWAYS
> - Baye's Theorem can be used to revise a probability analysis based on new information.
> - It is commonly used in finance to calculate or update risk evaluation.
> - And it is also used in other fields, such as analyses of medical test results.
<div align="center">
    <img src="https://i.postimg.cc/sx6YGXtc/data-science-bayes-theorem-2.png" alt="bayes_theorem" width="450">
</div>   
<br>   

## 🧪 Example 

Now let's see an simple example in the **medical test results** context about **Disease Testing**:  
A test for this disease is:

- 99% sensitive (true positive rate): If you **have** the disease, the test is positive 99% of the time.

- 95% specific (true negative rate): If you **don't** have the disease, the test is negative 95% of the time.

Now, if a person tests positive, what is the **probability they actually have the disease?**  

### 📊 Bayes' Theorem Formula
$$P(\text{Disease} \mid \text{Positive}) = \frac{P(\text{Positive} \mid \text{Disease}) \cdot P(\text{Disease})}{P(\text{Positive})}$$


### 🧮 Given:
- P(Disease) = 0.01
- P(No Disease) = 0.99
- P(Psitive | Disease) = 0.99
- P(Positive∣ No Disease) = 1−0.95 = 0.05
### 🧠 Step 1: Compute total probability of testing positive
P(Positive)=(0.99⋅0.01)+(0.05⋅0.99)=0.0099+0.0495=0.0594
### 🔍 Step 2: Apply Bayes' Theorem
$$P(\text{Disease} \mid \text{Positive}) = \frac{0.99 \times 0.01}{0.0594} = \frac{0.0099}{0.0594}$$

### ✅ Answer: About 16.7%
Even with a **positive test**, the chance of actually having the disease is only 16.7%, because the disease is rare and the test has some false positives.


### References
- <[Baye's Theorem](https://www.investopedia.com/terms/b/bayes-theorem.asp)>. Acess 29 may 2025 
