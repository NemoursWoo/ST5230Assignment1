import numpy as np
import time
from scipy.linalg import svd
from collections import Counter

def build_cooccurrence_matrix(sentences, window_size=5):
    """Builds a word co-occurrence matrix from sentences."""
    vocab = set(word for sentence in sentences for word in sentence)
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    
    cooccurrence_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    word_counts = Counter()

    for sentence in sentences:
        for i, word in enumerate(sentence):
            word_counts[word] += 1
            word_idx = word2idx[word]
            # Context window
            for j in range(max(0, i - window_size), min(len(sentence), i + window_size + 1)):
                if i != j:  # Ignore the word itself
                    context_word = sentence[j]
                    context_idx = word2idx[context_word]
                    cooccurrence_matrix[word_idx, context_idx] += 1

    return cooccurrence_matrix, word_counts, word2idx, idx2word

# Compute PMI matrix
def compute_pmi(cooccurrence_matrix, word_counts, idx2word, smoothing=1e-5):
    """Computes PMI matrix from co-occurrence data."""
    total_cooccurrences = cooccurrence_matrix.sum()
    word_freq = {word: count / total_cooccurrences for word, count in word_counts.items()}
    
    pmi_matrix = np.zeros_like(cooccurrence_matrix)
    print(pmi_matrix.shape[0], pmi_matrix.shape[1])

    for i in range(cooccurrence_matrix.shape[0]):
        for j in range(cooccurrence_matrix.shape[1]):
            if cooccurrence_matrix[i, j] > 0:
                p_wc = cooccurrence_matrix[i, j] / total_cooccurrences
                p_w = word_freq[idx2word[i]]
                p_c = word_freq[idx2word[j]]
                pmi_matrix[i, j] = max(np.log2((p_wc + smoothing) / (p_w * p_c + smoothing)), 0)  # Standard PMI
    print(pmi_matrix.shape[0], pmi_matrix.shape[1])
    return pmi_matrix

# Compute SPPMI matrix
def compute_sppmi(pmi_matrix, k=5):
    """Computes Shifted Positive PMI (SPPMI) by subtracting log(k) and clipping negatives."""
    sppmi_matrix = np.maximum(pmi_matrix - np.log2(k), 0)
    print(sppmi_matrix)
    return sppmi_matrix

def compute_low_rank_approximation(sppmi_matrix, rank=50):
    """Computes a low-rank approximation using truncated SVD: U_k sqrt(Sigma_k)."""
    U, S, _ = svd(sppmi_matrix, full_matrices=False)
    U_k = U[:, :rank]  # First `rank` components
    S_k = np.sqrt(np.diag(S[:rank]))  # Take sqrt of singular values
    return U_k @ S_k  # Low-rank approximation

# Train SPPMI-SVD model with training time
def train_sppmi_svd_model(sentences):
    start_time = time.time()
    cooccurrence_matrix, word_counts, _, idx2word = build_cooccurrence_matrix(sentences)
    pmi_matrix = compute_pmi(cooccurrence_matrix, word_counts, idx2word)
    sppmi_matrix = compute_sppmi(pmi_matrix)
    
    embeddings_sppmi_svd = compute_low_rank_approximation(sppmi_matrix, rank=50)
    sppmi_svd_time = time.time() - start_time
    return embeddings_sppmi_svd, sppmi_svd_time
