# from flask import Flask, render_template, request
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# import pickle
# import string

# app = Flask(__name__)

# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# def predict_spam(message):

#     # Preprocess the message 
#     transformed_sms = transform_text(message)
#     # Vectorize the preprocessed message 
#     vector_input = tfidf.transform([transformed_sms])
#     # Predict using ML model 
#     result = model.predict(vector_input)[0]
#     return result

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         input_sms = request.form['message']
#         result = predict_spam(input_sms)
#         return render_template('index.html', result=result)  # Pass 'result' to the template


# if __name__ == '__main__':
#     tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
#     model = pickle.load(open('model.pkl', 'rb'))
#     app.run(host='0.0.0.0')
# # localhost ip address 0.0.0.0



from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string  

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Load model and vectorizer globally
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    exit()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    text = [ps.stem(i) for i in text]

    return " ".join(text)

def predict_spam(message):
    try:
        transformed_sms = transform_text(message)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
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
