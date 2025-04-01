from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load GPT-2 model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Chat function to get GPT-2 responses
def chat_with_gpt2(user_input):
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Route to render the chatbot interface
@app.route("/")
def home():
    return render_template("chat.html")

# Route to handle chat input and response
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["user_input"]
    if user_input.lower() == "exit":
        return jsonify({"response": "Goodbye!"})
    response = chat_with_gpt2(user_input)
    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

