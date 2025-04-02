from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load GPT-2 model and tokenizer
model_name = "distilgpt2"
print("Loading model...")
try:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Chat function to get GPT-2 responses
def chat_with_gpt2(user_input):
    if model is None:
        return "Model is not loaded properly."
    
    try:
        inputs = tokenizer.encode(user_input, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_length=50, 
                num_return_sequences=1, 
                do_sample=True,  # Enables randomness
                top_k=50,  # Limits to top 50 words
                top_p=0.95  # Nucleus sampling
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated Response:", response)  # Debug print
        return response
    except Exception as e:
        print("Error during response generation:", e)
        return "An error occurred while generating a response."

# Route to render the chatbot interface
@app.route("/")
def home():
    return render_template("chat.html")

# Route to handle chat input and response
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input", "").strip()
    print("User Input:", user_input)  # Debug print
    
    if not user_input:
        return jsonify({"response": "I didn't receive any input."})
    
    if user_input.lower() == "exit":
        return jsonify({"response": "Goodbye!"})

    response = chat_with_gpt2(user_input)
    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
