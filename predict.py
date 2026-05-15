import joblib
import os

def load_model():
    model_path = os.path.join('models', 'toxic_comment_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
    
    print(f"Loading model from {model_path}...")
    return joblib.load(model_path)

def predict_toxicity(model, text):
    # The target columns we trained on
    target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Predict returns a 2D array, we take the first item
    prediction = model.predict([text])[0]
    print(f"Prediction: {prediction}")
    
    # Alternatively, you can use predict_proba for probabilities if you want to set custom thresholds
    # probabilities = model.predict_proba([text])
    # print(f"Probabilities: {probabilities}")
    
    print(f"\nAnalyzing comment: \"{text}\"")
    print("-" * 30)
    
    is_toxic = False
    for label, is_present in zip(target_columns, prediction):
        if is_present:
            is_toxic = True
            print(f"Flagged as: {label.upper()}")
            
    if not is_toxic:
        print("Comment appears to be clean.")
    print("-" * 30)

def main():
    try:
        model = load_model()
        print("\nModel loaded successfully! Type 'quit' or 'exit' to stop.")
        
        while True:
            user_input = input("\nEnter a comment to test: ")
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if not user_input.strip():
                continue
                
            predict_toxicity(model, user_input)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
