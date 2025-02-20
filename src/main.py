import pandas as pd
import numpy as np
import torch
import gensim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from . import skipgram
from . import SPPMI
from . import GloVe
from .data import dataloader as dl


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentences = dl.load_data()

skipgram_model, skipgram_time = skipgram.train_skipgram_model(sentences)
print(skipgram_time)

embeddings_sppmi_svd, sppmi_svd_time = SPPMI.train_sppmi_svd_model(sentences)
print(sppmi_svd_time)

glove_model, glove_time, word2idx = GloVe.train_glove_model(sentences)
print(glove_time)

sampled_items = list(word2idx.items())[:50]
sampled_words = [word for word, idx in sampled_items]
sampled_idx = [idx for word, idx in sampled_items]
sampled_idx_tensor = torch.tensor(sampled_idx).to(device)

embeddings_skipgram = skipgram_model.wv[sampled_words]
embeddings_glove = glove_model.embeddings(sampled_idx_tensor).detach().cpu().numpy()

# PCA transformation
pca = PCA(n_components=2)
skipgram_pca = pca.fit_transform(embeddings_skipgram)
SPPMI_pca = pca.fit_transform(embeddings_sppmi_svd)
glove_pca = pca.fit_transform(embeddings_glove)

# Function to plot PCA results separately
def plot_pca_results(pca_results, sampled_items, title, color, save_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(pca_results[:, 0], pca_results[:, 1], color=color)
    for word, idx in sampled_items:
        plt.annotate(word, (pca_results[idx, 0], pca_results[idx, 1]))
    plt.title(title)
    plt.savefig(save_path)
    plt.show()

# Plot Skip-gram PCA results
plot_pca_results(skipgram_pca, sampled_items, "PCA of Skip-gram Embeddings", "blue")

# Plot SPPMI-SVD PCA results
plot_pca_results(SPPMI_pca, sampled_items, "PCA of SPPMI-SVD Embeddings", "yellow", "SPPMI_SVD_PCA.png")

# Plot GloVe PCA results
plot_pca_results(glove_pca, sampled_items, "PCA of GloVe Embeddings", "red", "GloVe_PCA.png")

# # Cosine Similarity Evaluation for Specific Word Pairs
# word_pairs = [("diabetes", "insulin"), ("hypertension", "antibiotic"), ("surgery", "cbc")]
# similarity_scores = []

# for word1, word2 in word_pairs:
#     # Skip-gram similarity
#     if word1 in skipgram_model.wv and word2 in skipgram_model.wv:
#         sim_skipgram = cosine_similarity(
#             [skipgram_model.wv[word1]], [skipgram_model.wv[word2]]
#         )[0][0]
#     else:
#         sim_skipgram = None

#     # FastText similarity
#     if word1 in fasttext_model.wv and word2 in fasttext_model.wv:
#         sim_fasttext = cosine_similarity(
#             [fasttext_model.wv[word1]], [fasttext_model.wv[word2]]
#         )[0][0]
#     else:
#         sim_fasttext = None

#     similarity_scores.append((word1, word2, sim_skipgram, sim_fasttext))

# # Convert similarity results to DataFrame
# similarity_df = pd.DataFrame(similarity_scores, columns=["Word1", "Word2", "Skip-gram Cosine Similarity", "FastText Cosine Similarity"])
# print(similarity_df)
