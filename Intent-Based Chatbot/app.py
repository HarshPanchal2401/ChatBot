from flask import Flask, request, jsonify, render_template
import json
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

nltk.download('punkt')

# Load intents file
with open("intents.json", "r") as file:
    intents = json.load(file)

# Prepare data
def prepareData():
    patterns = []
    tags = []
    for intent in intents:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
    return patterns, tags
patterns, tags = prepareData()

# Clean data
def cleanData(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Vectorize data
def vectorize(patterns, user_input):
    vectorizer = TfidfVectorizer()
    patterns_cleaned = [cleanData(pattern) for pattern in patterns]
    user_input_cleaned = cleanData(user_input)
    X = vectorizer.fit_transform(patterns_cleaned + [user_input_cleaned])
    return X

app = Flask(__name__, template_folder=".")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

# Get response using NLP
def get_response_nlp(user_input):
    user_input = cleanData(user_input)
    logger.info(f"Cleaned User Input: {user_input}")
    
    try:
        X = vectorize(patterns, user_input)
        cosine_sim = cosine_similarity(X[-1], X[:-1])
        logger.info(f"Cosine Similarity: {cosine_sim}")
        max_similarity_index = cosine_sim.argmax()
        if cosine_sim[0][max_similarity_index] > 0.4:
            tag = tags[max_similarity_index]
            for intent in intents:
                if intent["tag"] == tag:
                    return random.choice(intent["responses"])
    except Exception as e:
        logger.error(f"Error in response generation: {e}")
    
    return "I'm sorry, I didn't understand that. Can you rephrase?"

# GUI for chatbot code
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/message", methods=["POST"])
def message():
    try:
        user_message = request.json.get("message")
        if not user_message:
            return jsonify({"response": "Please provide a message."}), 400
        
        logger.info(f"User Message: {user_message}")
        response = get_response_nlp(user_message)
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify({"response": "Something went wrong. Please try again later."}), 500


if __name__ == "__main__":
    app.run(debug=True)
