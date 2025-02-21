import numpy as np
import time
from scipy.linalg import svd
from collections import defaultdict
from . import utils
import torch

# Compute PMI matrix
def compute_pmi(cooccurrence_matrix, word_counts, idx2word, smoothing=1e-5):
    """Computes PMI matrix from sparse co-occurrence data."""
    total_cooccurrences = sum(cooccurrence_matrix.values())
    word_freq = {word: count / total_cooccurrences for word, count in word_counts.items()}
    
    pmi_matrix = defaultdict(float)  # Store PMI values sparsely
    
    for (i, j), count in cooccurrence_matrix.items():
        p_wc = count / total_cooccurrences
        p_w = word_freq[idx2word[i]]
        p_c = word_freq[idx2word[j]]
        pmi_value = max(np.log2((p_wc + smoothing) / (p_w * p_c + smoothing)), 0)  # Standard PMI
        pmi_matrix[(i, j)] = pmi_value
    
    return pmi_matrix

# Convert sparse matrix to dense matrix
def sparse_to_dense(sparse_matrix, vocab_size):
    """Converts a sparse dictionary-based matrix to a dense NumPy array."""
    dense_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for (i, j), value in sparse_matrix.items():
        dense_matrix[i, j] = value
    return dense_matrix

# Compute SPPMI matrix
def compute_sppmi(pmi_matrix_sparse, vocab_size, k=5):
    """Computes Shifted Positive PMI (SPPMI) by subtracting log(k) and clipping negatives."""
    pmi_matrix = sparse_to_dense(pmi_matrix_sparse, vocab_size)
    sppmi_matrix = np.maximum(pmi_matrix - np.log2(k), 0)
    print(sppmi_matrix)
    return sppmi_matrix

def compute_low_rank_approximation_cpu(sppmi_matrix, rank):
    """Computes a low-rank approximation using truncated SVD: U_k sqrt(Sigma_k)."""
    U, S, _ = svd(sppmi_matrix, full_matrices=False)
    U_k = U[:, :rank]  # First `rank` components
    S_k = np.sqrt(np.diag(S[:rank]))  # Take sqrt of singular values
    return U_k @ S_k  # Low-rank approximation

def compute_low_rank_approximation_gpu(sppmi_matrix, rank, device):
    """Computes a low-rank approximation using truncated SVD in PyTorch."""
    if not isinstance(sppmi_matrix, torch.Tensor): # Move to GPU
        sppmi_tensor = torch.tensor(sppmi_matrix, dtype=torch.float32, device=device)
    else:
        sppmi_tensor = sppmi_matrix.to(device)
    # Perform SVD
    U, S, _ = torch.linalg.svd(sppmi_tensor, full_matrices=False)
    # Keep only the top `rank` components
    U_k = U[:, :rank]
    S_k = torch.sqrt(torch.diag(S[:rank]))
    embeddings_gpu = U_k @ S_k  # Low-rank approximation
    return embeddings_gpu.cpu().numpy()  # Move back to CPU

def compute_low_rank_approximation(sppmi_matrix, rank, device):
    if device.type == "cuda":
        return compute_low_rank_approximation_gpu(sppmi_matrix, rank, device)
    else:
        return compute_low_rank_approximation_cpu(sppmi_matrix, rank)


# Train SPPMI-SVD model with training time
def train_sppmi_svd_model(sentences, device):
    start_time = time.time()
    cooccurrence_matrix, word_counts, _, idx2word = utils.build_cooccurrence_matrix(sentences, window_size=5)
    vocab_size = len(idx2word)
    pmi_matrix_sparse = compute_pmi(cooccurrence_matrix, word_counts, idx2word)
    sppmi_matrix = compute_sppmi(pmi_matrix_sparse, vocab_size)
    
    embeddings_sppmi_svd = compute_low_rank_approximation(sppmi_matrix, rank=50, device=device)
    sppmi_svd_time = time.time() - start_time
    return embeddings_sppmi_svd, sppmi_svd_time
