import pandas as pd
import joblib
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import os

def main():
    print("1. Loading dataset from Hugging Face...")
    # Loading the Jigsaw Toxicity dataset
    # This might take a bit of time to download the first time
    dataset = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge", split="train")

    # Convert to Pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    print("\nData Sample:")
    print(df.head())

    print("\n2. Preprocessing Data...")
    # The columns we want to predict
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Input text (X) and target labels (y)
    X = df['comment_text'].fillna(' ')
    y = df[target_columns]

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n3. Building the Model Pipeline...")
    # We use a Pipeline to combine TF-IDF and Logistic Regression
    # TF-IDF converts text into numerical features
    # MultiOutputClassifier allows LogisticRegression to predict multiple labels simultaneously
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000, class_weight='balanced')))
    ])

    print("\n4. Training the model (this might take a few minutes)...")
    pipeline.fit(X_train, y_train)

    print("\n5. Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_columns))

    print("\n6. Saving the model...")
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'toxic_comment_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Model successfully saved to {model_path}!")

if __name__ == "__main__":
    main()
