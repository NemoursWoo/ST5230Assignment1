import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import defaultdict


# define GloVe model
class GloVe(nn.Module):
    def __init__(self, vocab_size, embedding_dim, x_max=100, alpha=0.75):
        super(GloVe, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)
        
        # Weight initialization
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
        nn.init.zeros_(self.word_biases.weight)
        nn.init.zeros_(self.context_biases.weight)
        
        self.x_max = x_max
        self.alpha = alpha

    def forward(self, word_idx, context_idx, cooccurrence):
        word_vec = self.embeddings(word_idx)
        context_vec = self.context_embeddings(context_idx)
        word_bias = self.word_biases(word_idx).squeeze()
        context_bias = self.context_biases(context_idx).squeeze()

        log_cooccurrence = torch.log(1e-8 + cooccurrence.to_dense())  # Avoid log(0)
        weight = torch.pow(torch.clamp(cooccurrence / self.x_max, max=1), self.alpha)

        loss = weight * (torch.sum(word_vec * context_vec, dim=1) + word_bias + context_bias - log_cooccurrence) ** 2
        print(loss.mean())
        return loss.mean()

def build_cooccurrence_matrix(sentences, window_size=5):
    """Creates a word co-occurrence matrix from sentences."""
    vocab = set(word for sentence in sentences for word in sentence)
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    
    cooccurrence_matrix = defaultdict(float)
    
    for sentence in sentences:
        for i, word in enumerate(sentence):
            word_idx = word2idx[word]
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:
                    context_word = sentence[j]
                    context_idx = word2idx[context_word]
                    cooccurrence_matrix[(word_idx, context_idx)] += 1
    
    return cooccurrence_matrix, word2idx, idx2word    

# Train Glove model with training time
def train_glove(sentences, embedding_dim=50, window_size=5, epochs=100, lr=0.05):
    cooccurrence_matrix, _, word2idx, idx2word = build_cooccurrence_matrix(sentences, window_size)
    
    vocab_size = len(word2idx)
    model = GloVe(vocab_size, embedding_dim)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)  
    
    word_idx_list = [i for i in range(cooccurrence_matrix.shape[0])]
    context_idx_list = [j for j in range(cooccurrence_matrix.shape[1])]
    word_idx_tensor = torch.tensor(word_idx_list, dtype=torch.long)
    context_idx_tensor = torch.tensor(context_idx_list, dtype=torch.long)
    cooccurrence_tensor = torch.tensor(cooccurrence_matrix, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model(word_idx_tensor, context_idx_tensor, cooccurrence_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, word2idx, idx2word

def train_glove_model(sentences):
    start_time = time.time()
    glove_model, _, _ = train_glove(sentences)
    glove_time = time.time() - start_time
    return glove_model, glove_time