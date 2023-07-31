### NLP-text-classification
A template for an NLP text classification pipeline.
It can power conversational agents to understand user goals.
The model takes raw text queries as input and outputs a predicted intent label.
Performance is evaluated using stratified cross-validation.
**This is useful for goal-oriented chatbots and voice assistants to determine user intent and respond appropriately.**

 ❗Model needs improvement❗
  - English and non-English words including slang, abbreviations, typos, etc.
  - One of the CSV files  does not have enough samples for each class (in our case 5 pieces are required).
    
  🔴 Query_interpretation is an initial attempt to create a template for an NLP text classification. Score **(0.53)**
  🔴 Query_interpretation2 is using TF-IDF, but it didn't improve the score much.
  
### Overview
- CSV dataset with 'query' and 'intent' columns. Queries are text, intents are labels.
   - English and non-English words including slang, abbreviations, typos, etc.
   - One of the CSV files  does not have enough samples for each class (in our case 5 pieces is min required).
  
- Text preprocessing steps include lowercasing, removing punctuation, tokenizing into words, lemmatizing (grouping together different forms 
  of a word into a single base form), and removing stopwords.
  
- Label encoding is done.
- Added check that each intent class has at least 5 samples, otherwise StratifiedKFold can fail.

- A pipeline is created with CountVectorizer for feature extraction and LogisticRegression for the classifier.

- Grid search CV is used to tune hyperparameters:
  - ngram_range for the vectorizer
  - Regularization strength (C) for LogisticRegression

- Stratification is used in the CV split to handle class imbalance.

- The best parameters found are unigrams (ngram_range=1,1) and C=10.

- The best score is only 0.53, indicating the model is not very accurate. There is room for improvement.

### Analysis

The main result is the set of best hyperparameters and the best validation score after performing grid search cross-validation.

The best hyperparameters found were:
- vect__ngram_range = (1,1) 
- clf__C = 10

This means **unigrams** (individual words) worked better than bigrams (sequences of two consecutive words) for the CountVectorizer feature extraction. 

Regularization strength **C=10** worked best for the LogisticRegression classifier.
**Even though the regularization is weaker, overfitting is not happening here.**
The best validation score achieved was 0.5347826086956522.

- As I mentioned before, the score is quite low, only slightly better than random chance (0.5).
- We need to achieve results closer to 1.0.
- It is a good starting point.

### Ways to improve
- Try different classifiers like SVM, XGBoost, etc. 
- Optimize the text preprocessing - remove rare words, stemming, etc.
- Use TF-IDF instead of bag-of-words counts.
- Use pre-trained word embeddings as features.
- Try character n-grams instead of word n-grams.
- Use class weights to handle class imbalance.
  


  
 
