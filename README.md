# Emotion Detection from Text (NLP)

A text classification project that predicts the emotion expressed in a short sentence (e.g., sadness, anger, love). It compares Bag-of-Words and TF-IDF features with Naive Bayes and Logistic Regression, reaching **86.3% accuracy** with TF-IDF + Logistic Regression.

## Highlights
- 16,000 labeled English sentences across 6 emotion classes
- Full text-cleaning pipeline built with Python and NLTK
- Three model and feature combinations compared on the same train/test split
- Best model: **TF-IDF + Logistic Regression (86.3% accuracy)**

## Tech Stack
Python · Pandas · NumPy · NLTK · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Dataset
- File: `train.txt` (semicolon-separated: `text;emotion`)
- 16,000 rows, no missing values
- 6 emotion classes: sadness, anger, love, fear, joy, surprise
- Source: Kaggle "Emotions in Text" dataset

## Preprocessing
1. Convert text to lowercase
2. Remove punctuation
3. Remove numbers
4. Remove emojis and non-ASCII characters
5. Remove English stop words (NLTK)
6. Encode emotion labels as integers (0–5)
7. 80/20 train-test split (`random_state=42`)

## Feature Extraction and Models
| Features | Model | Accuracy |
|----------|-------|----------|
| Bag-of-Words (CountVectorizer) | Multinomial Naive Bayes | 76.8% |
| TF-IDF | Multinomial Naive Bayes | 66.1% |
| TF-IDF | Logistic Regression | **86.3%** |

**Takeaways**
- Logistic Regression clearly outperformed Naive Bayes on TF-IDF features, likely because it weights terms discriminatively instead of assuming word independence.
- Naive Bayes did better with raw counts than with TF-IDF here, which is worth knowing because MultinomialNB is designed around count-like features.


## Limitations
- Only the training file was used, split into train and test; a separate validation set would give a more reliable estimate.
- NLTK's stop-word list removes negations like "not" and "no", which can flip an emotion's meaning.
- Accuracy is the only metric reported so far.

## Future Improvements
- Keep negation words during preprocessing
- Add a confusion matrix and per-class precision, recall and F1
- Try SVM, Random Forest, or n-gram features
- Fine-tune a transformer such as DistilBERT
- Deploy as a small web app (FastAPI or Streamlit)

## Author
Mayuri Joshi
