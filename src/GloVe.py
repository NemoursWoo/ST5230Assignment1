import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, TensorDataset
from . import utils 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define GloVe model
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

    def forward(self, word_idx, context_idx):
        word_idx, context_idx = word_idx.to(device), context_idx.to(device)
        word_vec = self.embeddings(word_idx)
        context_vec = self.context_embeddings(context_idx)
        word_bias = self.word_biases(word_idx).view(-1, 1)
        context_bias = self.context_biases(context_idx).view(-1, 1)

        prediction = torch.sum(word_vec * context_vec, dim=1, keepdim=True) + word_bias + context_bias
        return prediction


# Define loss function
def glove_loss(prediction, cooccurrence, x_max=100, alpha=0.75):
    log_cooccurrence = torch.log(1e-8 + cooccurrence)  # Avoid log(0)
    weight = torch.pow(torch.clamp(cooccurrence / x_max, max=1), alpha).view(-1, 1)
    return torch.mean(weight * (prediction - log_cooccurrence) ** 2)


# Training function
def train_glove(sentences, embedding_dim=50, window_size=5, epochs=10, lr=0.05, batch_size=1024):
    """Trains the GloVe model."""
    # Build co-occurrence matrix
    cooccurrence_matrix, _, word2idx, idx2word = utils.build_cooccurrence_matrix(sentences, window_size)
    
    vocab_size = len(word2idx)
    model = GloVe(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)  
    
    # Convert co-occurrence matrix into dataset
    word_idx_list, context_idx_list, cooccurrence_list = zip(*[
        (word_idx, context_idx, cooccurrence_matrix[(word_idx, context_idx)])
        for (word_idx, context_idx) in cooccurrence_matrix
    ])
    
    # Convert lists to tensors and move to device
    word_idx_tensor = torch.tensor(word_idx_list, dtype=torch.long).to(device)
    context_idx_tensor = torch.tensor(context_idx_list, dtype=torch.long).to(device)
    cooccurrence_tensor = torch.tensor(cooccurrence_list, dtype=torch.float32).view(-1, 1).to(device)

    # Create DataLoader for mini-batch training
    dataset = TensorDataset(word_idx_tensor, context_idx_tensor, cooccurrence_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for word_batch, context_batch, cooccurrence_batch in dataloader:
            optimizer.zero_grad()
            prediction = model(word_batch, context_batch)  # Get predictions
            loss = glove_loss(prediction, cooccurrence_batch)  # Compute loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {epoch_loss / len(dataloader)}")

    return model, word2idx, idx2word


# Train GloVe model with training time
def train_glove_model(sentences):
    """Trains the GloVe model and records the training time."""
    start_time = time.time()
    glove_model, word2idx, _ = train_glove(sentences)
    glove_time = time.time() - start_time
    return glove_model, glove_time, word2idx

