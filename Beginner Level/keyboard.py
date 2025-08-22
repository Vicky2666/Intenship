import numpy as np
import pandas as pd
import re
from collections import defaultdict, Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gradio as gr
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Sample training data - in a real application, you'd use a much larger corpus
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "a journey of a thousand miles begins with a single step",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "the early bird catches the worm",
    "better late than never",
    "actions speak louder than words",
    "beauty is in the eye of the beholder",
    "don't count your chickens before they hatch",
    "every cloud has a silver lining",
    "fortune favors the bold",
    "good things come to those who wait",
    "honesty is the best policy",
    "if it ain't broke don't fix it",
    "keep your friends close and your enemies closer",
    "knowledge is power",
    "look before you leap",
    "money doesn't grow on trees",
    "no pain no gain",
    "practice makes perfect",
    "the best of both worlds",
    "the pen is mightier than the sword",
    "time is money",
    "when in Rome do as the Romans do",
    "you can't judge a book by its cover"
]


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text


# Preprocess the corpus
processed_corpus = [preprocess_text(text) for text in corpus]


# Build n-gram model
class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.start_token = "<s>"
        self.end_token = "</s>"

    def train(self, corpus):
        for sentence in corpus:
            tokens = sentence.split()
            tokens = [self.start_token] * (self.n - 1) + tokens + [self.end_token]
            self.vocab.update(tokens)

            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n - 1])
                next_word = tokens[i + self.n - 1]
                self.ngrams[ngram][next_word] += 1

    def predict_next_word(self, context, k=3):
        context = preprocess_text(context)
        tokens = context.split()

        if len(tokens) < self.n - 1:
            tokens = [self.start_token] * (self.n - 1 - len(tokens)) + tokens

        ngram = tuple(tokens[-(self.n - 1):])

        if ngram in self.ngrams:
            predictions = self.ngrams[ngram].most_common(k)
            return [word for word, count in predictions if word != self.end_token]

        # Backoff to lower n-gram if needed
        if self.n > 2:
            backoff_model = NGramModel(self.n - 1)
            backoff_model.train(processed_corpus)
            return backoff_model.predict_next_word(context, k)

        return ["the", "and", "to"]  # Default common words


# Neural network model for next word prediction
class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=1):
        super(NextWordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])  # Get the last time step
        return output


# Prepare data for neural network
all_text = " ".join(processed_corpus)
words = all_text.split()
vocab = sorted(set(words))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

# Create training sequences
sequence_length = 5
sequences = []
next_words = []

for i in range(len(words) - sequence_length):
    seq = words[i:i + sequence_length]
    next_word = words[i + sequence_length]
    sequences.append([word_to_idx[word] for word in seq])
    next_words.append(word_to_idx[next_word])

# Convert to tensors
X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(next_words, dtype=torch.long)


# Create dataset and dataloader
class TextDataset(Dataset):
    def __init__(self, sequences, next_words):
        self.sequences = sequences
        self.next_words = next_words

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.next_words[idx]


dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NextWordPredictor(vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")


# Function to predict next word using neural network
def neural_predict_next_word(context, k=3):
    context = preprocess_text(context)
    tokens = context.split()

    if len(tokens) < sequence_length:
        tokens = ["<unk>"] * (sequence_length - len(tokens)) + tokens
    else:
        tokens = tokens[-sequence_length:]

    # Convert to indices
    seq_indices = [word_to_idx.get(word, word_to_idx.get("<unk>", 0)) for word in tokens]
    seq_tensor = torch.tensor([seq_indices], dtype=torch.long).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(seq_tensor)
        probabilities = torch.softmax(output, dim=1)
        top_k = torch.topk(probabilities, k, dim=1)

    # Convert indices to words
    top_words = [idx_to_word[idx.item()] for idx in top_k.indices[0]]
    top_probs = [f"{prob.item():.2f}" for prob in top_k.values[0]]

    return list(zip(top_words, top_probs))


# Train n-gram model
ngram_model = NGramModel(n=3)
ngram_model.train(processed_corpus)

# Autocorrect function using cosine similarity
vectorizer = CountVectorizer().fit([" ".join(vocab)])
vocab_vectors = vectorizer.transform(vocab)


def autocorrect(word):
    if not word or word in vocab:
        return word

    word_vec = vectorizer.transform([word])
    similarities = cosine_similarity(word_vec, vocab_vectors)
    most_similar_idx = np.argmax(similarities)
    return vocab[most_similar_idx]


# Combined prediction function
def predict_next_word(context, method="ngram", k=3):
    if method == "ngram":
        return ngram_model.predict_next_word(context, k)
    else:
        predictions = neural_predict_next_word(context, k)
        return [f"{word} ({prob})" for word, prob in predictions]


# Gradio interface
def autocorrect_and_suggest(text, method):
    # If text is empty, return empty suggestions
    if not text.strip():
        return text, "", "", ""

    # Autocorrect the last word
    words = text.split()
    if words:
        last_word = words[-1]
        corrected = autocorrect(last_word)
        if corrected != last_word:
            words[-1] = corrected
            text = " ".join(words)

    # Get predictions
    predictions = predict_next_word(text, method, 3)

    # Return the corrected text and predictions
    return text, predictions[0] if len(predictions) > 0 else "", predictions[1] if len(predictions) > 1 else "", \
    predictions[2] if len(predictions) > 2 else ""


# Create Gradio interface
with gr.Blocks(title="Autocorrect Keyboard with Next Word Prediction") as demo:
    gr.Markdown("# 🎯 Autocorrect Keyboard with Next Word Prediction")
    gr.Markdown("Type a sentence and see the next word predictions. Your text will be automatically corrected.")

    with gr.Row():
        method = gr.Radio(choices=["ngram", "neural"], value="ngram", label="Prediction Method")

    with gr.Row():
        input_text = gr.Textbox(
            label="Type your text here",
            placeholder="Start typing...",
            lines=2,
            elem_id="input-text"
        )

    with gr.Row():
        corrected_text = gr.Textbox(
            label="Autocorrected text",
            interactive=False
        )

    gr.Markdown("### Next Word Suggestions")

    with gr.Row():
        suggestion1 = gr.Button("Suggestion 1", elem_id="suggestion-1")
        suggestion2 = gr.Button("Suggestion 2", elem_id="suggestion-2")
        suggestion3 = gr.Button("Suggestion 3", elem_id="suggestion-3")

    # Set up event handlers
    input_text.change(
        fn=autocorrect_and_suggest,
        inputs=[input_text, method],
        outputs=[corrected_text, suggestion1, suggestion2, suggestion3]
    )

    method.change(
        fn=autocorrect_and_suggest,
        inputs=[input_text, method],
        outputs=[corrected_text, suggestion1, suggestion2, suggestion3]
    )


    # Function to handle suggestion clicks
    def add_suggestion(text, suggestion):
        return text + " " + suggestion.split(" (")[0]  # Remove probability if present


    suggestion1.click(
        fn=add_suggestion,
        inputs=[input_text, suggestion1],
        outputs=input_text
    )

    suggestion2.click(
        fn=add_suggestion,
        inputs=[input_text, suggestion2],
        outputs=input_text
    )

    suggestion3.click(
        fn=add_suggestion,
        inputs=[input_text, suggestion3],
        outputs=input_text
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()