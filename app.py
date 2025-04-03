from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

# Initialize Flask app
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
    model = None  # Prevent crashes if model fails to load

# Function to generate responses
def chat_with_gpt2(user_input):
    if model is None:
        return "Model is not loaded properly."

    try:
        # Tokenize and encode the input
        inputs = tokenizer.encode(user_input, return_tensors="pt", add_special_tokens=True)

        # Generate output with sampling
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=50,  # Limit response length
                num_return_sequences=1,
                do_sample=True,  # Enables randomness
                top_k=50,  # Limits to top 50 words
                top_p=0.95,  # Nucleus sampling
                temperature=0.7  # Controls response variation
            )

        # Decode the response back to text
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error during response generation: {e}")
        return "An error occurred while generating a response."

# Route to render the chatbot interface
@app.route("/")
def home():
    return render_template("chat.html")

# Route to handle chat input and response
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input", "").strip()

    if not user_input:
        return jsonify({"response": "I didn't receive any input."})

    if user_input.lower() == "exit":
        return jsonify({"response": "Goodbye!"})    

    response = chat_with_gpt2(user_input)
    return jsonify({"response": response})

# Run Flask app on Render-assigned port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port, debug=False)  # Set debug=False for production
