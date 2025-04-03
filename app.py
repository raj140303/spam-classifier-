from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string

app = Flask(__name__)

# Download required NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Load the TF-IDF vectorizer and model globally
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    exit()

def transform_text(text):
    """
    Convert input text to lowercase, tokenize it, filter out non-alphanumeric tokens,
    remove stopwords and punctuation, and apply stemming.
    """
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    # Filter out non-alphanumeric tokens
    filtered_tokens = [token for token in tokens if token.isalnum()]
    
    # Remove stopwords and punctuation
    cleaned_tokens = [
        token for token in filtered_tokens
        if token not in stopwords.words('english') and token not in string.punctuation
    ]
    
    # Apply stemming
    stemmed_tokens = [ps.stem(token) for token in cleaned_tokens]
    
    return " ".join(stemmed_tokens)

def predict_spam(message):
    """
    Transforms the input message and predicts whether it is spam using the loaded model.
    This updated version uses predict_proba (if available) to set a threshold.
    """
    try:
        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms])
        
        # Debugging output
        print("DEBUG: Original message:", message)
        print("DEBUG: Transformed message:", transformed_sms)
        print("DEBUG: Vector input shape:", vector_input.shape)
        
        # Use predict_proba if available for getting confidence scores.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector_input)[0]
            # Assuming class 1 corresponds to spam
            spam_prob = probabilities[1]
            print("DEBUG: Predicted probabilities:", probabilities)
            print("DEBUG: Spam probability:", spam_prob)
            
            threshold = 0.5  # You can adjust this threshold as needed.
            result = "spam" if spam_prob > threshold else "not spam"
        else:
            # Fallback: use the model's predict result directly.
            pred = model.predict(vector_input)[0]
            print("DEBUG: Prediction output:", pred)
            result = "spam" if pred == 1 else "not spam"
        
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_sms = request.form['message']
        result = predict_spam(input_sms)
        return render_template('index.html', result=result)
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
