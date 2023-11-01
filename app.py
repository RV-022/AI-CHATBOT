from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import spacy
import torch.nn as nn
import torch.optim as optim


nlp = spacy.load("en_core_web_sm")
with open("modified_dataset.csv", "r") as file:
    words = file.read().split()
    unique_words = set(words)

vocabulary = {"<PAD>": 0, "<UNK>": 1}  # Initialize with special tokens
with open("modified_dataset.csv", "r") as file:
    for line in file:
        words = line.strip().split()  # Split by whitespace, adapt to your data format
        for word in words:
            if word not in vocabulary:
                vocabulary[word] = len(vocabulary)


def tokenize_and_convert_to_tensor(text, vocabulary, output_size):
    # Tokenize the text using spaCy
    tokens = [token.text for token in nlp(text)]

    # Convert tokens to indices using the provided vocabulary
    indices = [vocabulary.get(token, vocabulary["<UNK>"]) for token in tokens]

    # Ensure indices are within the valid range
    indices = [min(idx, output_size - 1) for idx in indices]

    # Convert the list of indices to a PyTorch tensor
    tensor = torch.LongTensor(indices)

    return tensor

class SimpleChatbot(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleChatbot, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        output = self.out(output)
        return output, hidden


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chatbot.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


if __name__ == '__main__':
    app.run()
