# NLP-text-classification
A template for an NLP text classification pipeline using scikit-learn. 
The model takes raw text queries as input and outputs a predicted intent label.
Performance is evaluated using stratified cross-validation.
**This is useful for goal-oriented chatbots and voice assistants to determine user intent and respond appropriately.**

- Initial attempt to create a template for an NLP text classification.
- Model needs improvement as its score is still pretty low. **(0.53)**
  
  
  English and non-English words including slang, abbreviations, typos, etc.
  One of the CSV files  does not have enough samples for each class (in our case 5 pieces is min required).

# Overview
- CSV dataset with 'query' and 'intent' columns. Queries are text, intents are labels.
  English and non-English words including slang, abbreviations, typos, etc.
  One of the CSV files  does not have enough samples for each class (in our case 5 pieces is min required).
  
- Text preprocessing steps include lowercasing, removing punctuation,tokenizing into words, lemmatizing (grouping together different forms 
  of a word into a single base form), and removing stopwords.
  
- Label encoding is done on the intents to numeric labels.
- Added check that each intent class has at least 5 samples, otherwise StratifiedKFold can fail.

- A pipeline is created with CountVectorizer for feature extraction and LogisticRegression for the classifier.

- Grid search CV is used to tune hyperparameters:
  - ngram_range for the vectorizer
  - Regularization strength (C) for LogisticRegression

- Stratification is used in the CV split to handle class imbalance.

- The best parameters found are unigrams (ngram_range=1,1) and C=10.

- The best score is only 0.53, indicating the model is not very accurate. There is room for improvement.

# Ways to improve
- Try different classifiers like SVM, XGBoost, etc. 

- Optimize the text preprocessing - remove rare words, stemming, etc.

- Use TF-IDF instead of bag-of-words counts.

- Use pre-trained word embeddings as features.

- Try character n-grams instead of word n-grams.

- Use class weights to handle class imbalance.


  
 
