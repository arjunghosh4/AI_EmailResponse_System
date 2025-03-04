from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from email.message import EmailMessage
import smtplib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

import pandas as pd
df = pd.read_csv("SupportDataset.csv")
df.dropna(subset=["subject", "body", "answer"], inplace=True)

df["text"] = df["subject"] + " " + df["body"]

# Store question-answer pairs
qa_pairs = list(zip(df["text"], df["answer"]))

# Convert all past questions into a vector format
vectorizer = TfidfVectorizer(stop_words="english")
question_vectors = vectorizer.fit_transform([q for q, _ in qa_pairs])

def find_best_past_answer(user_question):
    user_vector = vectorizer.transform([user_question])  # Convert user input into vector
    similarities = cosine_similarity(user_vector, question_vectors)  # Compare with past questions
    best_match_index = similarities.argmax()  # Get the index of the best match
    return qa_pairs[best_match_index][1]  # Return the corresponding past answer

def generate_response(subject, body):
    user_input = subject + " " + body  # Combine user input

    # Retrieve the closest past answer
    best_answer = find_best_past_answer(user_input)

    return best_answer if best_answer else "Thank you for your email. Our team will get back to you soon."


# Initialize Flask app
app = Flask(__name__)

# Detect device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load Model and Tokenizer Correctly
model = BertForSequenceClassification.from_pretrained("my_model").to(device)
tokenizer = BertTokenizer.from_pretrained("my_tokenizer")

# ✅ Reverse label mapping
label_map = {
    "Billing and Payments": 0,
    "Customer Service": 1,
    "General Inquiry": 2,
    "Human Resources": 3,
    "IT Support": 4,
    "Product Support": 5,
    "Returns and Exchanges": 6,
    "Sales and Pre-Sales": 7,
    "Service Outages and Maintenance": 8,
    "Technical Support": 9
}
reverse_label_map = {v: k for k, v in label_map.items()}

# ✅ Function to classify emails
def classify_email(subject, body):
    text = subject + " " + body
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, axis=-1).item()
    category = reverse_label_map[predicted_class]
    return category

def generate_response(subject, body):
    user_input = subject + " " + body  # Combine user input
    best_answer = find_best_past_answer(user_input)  # Retrieve past answer
    return best_answer if best_answer else "Thank you for your email. Our team will get back to you soon."

# ✅ API Endpoint to handle incoming emails
@app.route("/send_email", methods=["POST"])
def handle_email():
    data = request.json
    customer_email = data.get("customer_email")
    email_subject = data.get("subject")
    email_body = data.get("body")

    # Step 1: Classify email (Optional, if still needed)
    category = classify_email(email_subject, email_body)

    # Step 2: Get AI-generated response using past answers
    response = generate_response(email_subject, email_body)

    return jsonify({
        "customer_email": customer_email,
        "predicted_category": category,
        "response": response
    })

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Serve the HTML form

# ✅ Run Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
