# Reflection Paper: Spam Detection
## February 22, 2026

## 1. Implementation Process

I loaded the mail dataset with Category (spam=0, ham=1) and Message columns. I cleaned the data by replacing NaN values with empty strings and encoded the labels numerically.

I used TfidfVectorizer to convert text messages into numerical features. TF-IDF measures word importance by considering both term frequency and how rare the word is across all documents. I used an 80/20 train-test split with random_state=42.

I trained three models: Logistic Regression, Random Forest (200 trees), and Naive Bayes (MultinomialNB). All models were evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

## 2. Model Comparison

Random Forest achieved the highest accuracy (98.3%) and best recall (87.2%), meaning it caught more spam messages. Naive Bayes had good accuracy (97.7%) with 82.6% recall. Logistic Regression had the lowest recall (75.8%) but still performed well overall.

For spam detection, recall is critical because missing spam (false negatives) is worse than flagging legitimate emails (false positives). Random Forest's higher recall makes it the best choice for this task.

## 3. Naive Bayes Explanation

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It calculates the probability that a message is spam given the words it contains. The "naive" part means it assumes all words are independent of each other.

MultinomialNB works well for text classification because it handles word counts naturally. It's fast to train and works well even with small datasets. Despite the independence assumption being unrealistic, Naive Bayes often performs surprisingly well for spam detection.

## 4. Metrics Discussion

- Accuracy: Overall correctness. Good for balanced datasets.
- Precision: Of predicted spam, how many are actually spam. High precision means fewer false alarms.
- Recall: Of actual spam, how many did we catch. High recall means fewer missed spam.
- F1-Score: Balance between precision and recall. Useful when both matter.
- Confusion Matrix: Shows true positives, false positives, true negatives, false negatives.

For spam detection, recall is most important because users want to catch all spam. Precision matters too since false positives annoy users. F1-Score helps balance both concerns.

## 5. Findings

All three models performed well (96-98% accuracy), showing that TF-IDF features work effectively for spam detection. Random Forest achieved the best overall performance with highest accuracy and recall.

The sanity checks showed models can disagree on edge cases. Some promotional messages are hard to classify because they use legitimate language. Random Forest's ensemble approach helps it handle these ambiguous cases better than single models.

For production spam filters, Random Forest is recommended due to its superior recall and accuracy. However, Naive Bayes could be used when speed and simplicity are priorities.
