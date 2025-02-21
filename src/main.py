import pandas as pd
import numpy as np
import torch
import random
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
from . import utils
from .data import dataloader as dl



# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentences = dl.load_data()
vocab = set(word for sentence in sentences for word in sentence)
print(len(vocab))
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

skipgram_model, skipgram_time = skipgram.train_skipgram_model(sentences)
print(skipgram_time)

# embeddings_sppmi_svd, sppmi_svd_time = SPPMI.train_sppmi_svd_model(sentences)
# print(sppmi_svd_time)

# glove_model, glove_time, word2idx = GloVe.train_glove_model(sentences)
# print(glove_time)

# Get keywords
keywords = utils.get_keywords()

# Filter word2idx to keep only words that exist in keywords
filtered_word2idx = {word: idx for word, idx in word2idx.items() if word in keywords}
if len(filtered_word2idx) > 50:
    sampled_word2idx = dict(random.sample(list(filtered_word2idx.items()), 50))  # Select 50 random items
else:
    sampled_word2idx = filtered_word2idx  # Keep all if <= 50

sampled_words = [word for word, idx in sampled_word2idx.items()]
print(sampled_words)
sampled_idx = [idx for word, idx in sampled_word2idx.items()]
sampled_idx_tensor = torch.tensor(sampled_idx, dtype=torch.long, device=device)

# sampled_idx_array = np.array(sampled_idx)
# new_embeddings_sppmi_svd = embeddings_sppmi_svd[sampled_idx_array]

embeddings_skipgram = skipgram_model.wv[sampled_words]
# embeddings_glove = glove_model.embeddings(sampled_idx_tensor).detach().cpu().numpy()

# print(embeddings_glove)

# PCA transformation
pca = PCA(n_components=2)
skipgram_pca = pca.fit_transform(embeddings_skipgram)
# SPPMI_pca = pca.fit_transform(new_embeddings_sppmi_svd)
# glove_pca = pca.fit_transform(embeddings_glove)

# Function to plot PCA results separately
def plot_pca_results(pca_results, sampled_words, title, color, save_path):
    plt.figure(figsize=(6, 5))
    plt.scatter(pca_results[:, 0], pca_results[:, 1], color=color)
    for i, word in enumerate(sampled_words):
        plt.annotate(word, (pca_results[i, 0], pca_results[i, 1]))

    plt.title(title)
    plt.savefig(save_path)
    plt.show()

# Plot Skip-gram PCA results
plot_pca_results(skipgram_pca, sampled_words, "PCA of Skip-gram Embeddings", "blue", "skip_gram_PCA.png")

# # Plot SPPMI-SVD PCA results
# plot_pca_results(SPPMI_pca, sampled_words, "PCA of SPPMI-SVD Embeddings", "yellow", "SPPMI_SVD_PCA.png")

# # Plot GloVe PCA results
# plot_pca_results(glove_pca, sampled_words, "PCA of GloVe Embeddings", "red", "GloVe_PCA.png")



# To do cosine similarity evaluation, we decide to use two types of word pairs: both words from the category Diseases and each word from categories Medications and Procedures separately.
words_diseases = [word for word in sampled_words if keywords.get(word) == "Diseases"]
words_medications = [word for word in sampled_words if keywords.get(word) == "Medications"]
words_procedures = [word for word in sampled_words if keywords.get(word) == "Procedures"]
words_lab_tests = [word for word in sampled_words if keywords.get(word) == "Laboratory_tests"]

def generate_word_pairs(words1, words2):
    return list(itertools.product(words1, words2))
    
diseases_word_pairs = generate_word_pairs(words_diseases, words_diseases)
medications_word_pairs = generate_word_pairs(words_medications, words_medications)
procedures_word_pairs = generate_word_pairs(words_procedures, words_procedures)
lab_tests_word_pairs = generate_word_pairs(words_lab_tests, words_lab_tests)
diseases_medications_word_pairs = generate_word_pairs(words_diseases, words_medications)
diseases_procedures_word_pairs = generate_word_pairs(words_diseases, words_procedures)
diseases_lab_tests_word_pairs = generate_word_pairs(words_diseases, words_lab_tests)
medications_procedures_word_pairs = generate_word_pairs(words_medications, words_procedures)
medications_lab_tests_word_pairs = generate_word_pairs(words_medications, words_lab_tests)
procedures_lab_tests_word_pairs = generate_word_pairs(words_procedures, words_lab_tests)

def calculate_cosine_similarity_skipgram(word_pairs, skipgram_model):
    similarity_scores = []
    # Skip-gram similarity
    for word1, word2 in word_pairs:
        sim = cosine_similarity(
            [skipgram_model.wv[word1]], [skipgram_model.wv[word2]]
        )[0][0]
        similarity_scores.append((word1, word2, sim))
    return similarity_scores

def calculate_cosine_similarity_sppmi_svd(word_pairs, word2idx, embeddings_sppmi_svd):
    similarity_scores = []
    # SPPMI-SVD similarity
    for word1, word2 in word_pairs:
        idx1, idx2 = word2idx[word1], word2idx[word2]
        sim = cosine_similarity(
            [embeddings_sppmi_svd[idx1]], [embeddings_sppmi_svd[idx2]]
        )[0][0]
        similarity_scores.append((word1, word2, sim))
    return similarity_scores

def calculate_cosine_similarity_glove(word_pairs, word2idx, glove_model):
    similarity_scores = []
    # GloVe similarity
    for word1, word2 in word_pairs:
        idx1, idx2 = word2idx[word1], word2idx[word2]
        idx1_tensor = torch.tensor([idx1], dtype=torch.long, device=device)
        idx2_tensor = torch.tensor([idx2], dtype=torch.long, device=device)
        sim = cosine_similarity(
            [glove_model.embeddings(idx1_tensor).detach().cpu().numpy()], [glove_model.embeddings(idx2_tensor).detach().cpu().numpy()]
        )[0][0]
        similarity_scores.append((word1, word2, sim))
    return similarity_scores

# concactenate all similarity scores of skipgram
similarity_scores_diseases_skipgram = calculate_cosine_similarity_skipgram(diseases_word_pairs, skipgram_model)
similarity_scores_medications_skipgram = calculate_cosine_similarity_skipgram(medications_word_pairs, skipgram_model)
similarity_scores_procedures_skipgram = calculate_cosine_similarity_skipgram(procedures_word_pairs, skipgram_model)
similarity_scores_lab_tests_skipgram = calculate_cosine_similarity_skipgram(lab_tests_word_pairs, skipgram_model)
similarity_scores_diseases_medications_skipgram = calculate_cosine_similarity_skipgram(diseases_medications_word_pairs, skipgram_model)
similarity_scores_diseases_procedures_skipgram = calculate_cosine_similarity_skipgram(diseases_procedures_word_pairs, skipgram_model)
similarity_scores_diseases_lab_tests_skipgram = calculate_cosine_similarity_skipgram(diseases_lab_tests_word_pairs, skipgram_model)
similarity_scores_medications_procedures_skipgram = calculate_cosine_similarity_skipgram(medications_procedures_word_pairs, skipgram_model)
similarity_scores_medications_lab_tests_skipgram = calculate_cosine_similarity_skipgram(medications_lab_tests_word_pairs, skipgram_model)
similarity_scores_procedures_lab_tests_skipgram = calculate_cosine_similarity_skipgram(procedures_lab_tests_word_pairs, skipgram_model)
similarity_scores_skipgram = similarity_scores_diseases_skipgram + similarity_scores_medications_skipgram + similarity_scores_procedures_skipgram + similarity_scores_lab_tests_skipgram + similarity_scores_diseases_medications_skipgram + similarity_scores_diseases_procedures_skipgram + similarity_scores_diseases_lab_tests_skipgram + similarity_scores_medications_procedures_skipgram + similarity_scores_medications_lab_tests_skipgram + similarity_scores_procedures_lab_tests_skipgram

## concactenate all similarity scores of sppmi_svd
# similarity_scores_diseases_sppmi_svd = calculate_cosine_similarity_sppmi_svd(diseases_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_medications_sppmi_svd = calculate_cosine_similarity_sppmi_svd(medications_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_procedures_sppmi_svd = calculate_cosine_similarity_sppmi_svd(procedures_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_lab_tests_sppmi_svd = calculate_cosine_similarity_sppmi_svd(lab_tests_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_diseases_medications_sppmi_svd = calculate_cosine_similarity_sppmi_svd(diseases_medications_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_diseases_procedures_sppmi_svd = calculate_cosine_similarity_sppmi_svd(diseases_procedures_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_diseases_lab_tests_sppmi_svd = calculate_cosine_similarity_sppmi_svd(diseases_lab_tests_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_medications_procedures_sppmi_svd = calculate_cosine_similarity_sppmi_svd(medications_procedures_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_medications_lab_tests_sppmi_svd = calculate_cosine_similarity_sppmi_svd(medications_lab_tests_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_procedures_lab_tests_sppmi_svd = calculate_cosine_similarity_sppmi_svd(procedures_lab_tests_word_pairs, word2idx, embeddings_sppmi_svd)
# similarity_scores_sppmi_svd = similarity_scores_diseases_sppmi_svd + similarity_scores_medications_sppmi_svd + similarity_scores_procedures_sppmi_svd + similarity_scores_lab_tests_sppmi_svd + similarity_scores_diseases_medications_sppmi_svd + similarity_scores_diseases_procedures_sppmi_svd + similarity_scores_diseases_lab_tests_sppmi_svd + similarity_scores_medications_procedures_sppmi_svd + similarity_scores_medications_lab_tests_sppmi_svd + similarity_scores_procedures_lab_tests_sppmi_svd

## concactenate all similarity scores of glove
# similarity_scores_diseases_glove = calculate_cosine_similarity_glove(diseases_word_pairs, word2idx, glove_model)
# similarity_scores_medications_glove = calculate_cosine_similarity_glove(medications_word_pairs, word2idx, glove_model)
# similarity_scores_procedures_glove = calculate_cosine_similarity_glove(procedures_word_pairs, word2idx, glove_model)
# similarity_scores_lab_tests_glove = calculate_cosine_similarity_glove(lab_tests_word_pairs, word2idx, glove_model)
# similarity_scores_diseases_medications_glove = calculate_cosine_similarity_glove(diseases_medications_word_pairs, word2idx, glove_model)
# similarity_scores_diseases_procedures_glove = calculate_cosine_similarity_glove(diseases_procedures_word_pairs, word2idx, glove_model)
# similarity_scores_diseases_lab_tests_glove = calculate_cosine_similarity_glove(diseases_lab_tests_word_pairs, word2idx, glove_model)
# similarity_scores_medications_procedures_glove = calculate_cosine_similarity_glove(medications_procedures_word_pairs, word2idx, glove_model)
# similarity_scores_medications_lab_tests_glove = calculate_cosine_similarity_glove(medications_lab_tests_word_pairs, word2idx, glove_model)
# similarity_scores_procedures_lab_tests_glove = calculate_cosine_similarity_glove(procedures_lab_tests_word_pairs, word2idx, glove_model)
# similarity_scores_glove = similarity_scores_diseases_glove + similarity_scores_medications_glove + similarity_scores_procedures_glove + similarity_scores_lab_tests_glove + similarity_scores_diseases_medications_glove + similarity_scores_diseases_procedures_glove + similarity_scores_diseases_lab_tests_glove + similarity_scores_medications_procedures_glove + similarity_scores_medications_lab_tests_glove + similarity_scores_procedures_lab_tests_glove

# Convert similarity results to DataFrame
def generate_similarity_df(similarity_scores, model_name):
    similarity_df = pd.DataFrame(similarity_scores, columns=["Word1", "Word2", f"{model_name} Cosine Similarity"])
    return similarity_df

similarity_df_skipgram = generate_similarity_df(similarity_scores_skipgram, "Skip-gram")
print(similarity_df_skipgram)
# similarity_df_sppmi_svd = generate_similarity_df(similarity_scores_sppmi_svd, "SPPMI-SVD")
# print(similarity_df_sppmi_svd)
# similarity_df_glove = generate_similarity_df(similarity_scores_glove, "GloVe")
# print(similarity_df_glove)

similarity_df_skipgram.to_csv("skipgram_similarity.csv", index=False)
# similarity_df_sppmi_svd.to_csv("sppmi_svd_similarity.csv", index=False)
# similarity_df_glove.to_csv("glove_similarity.csv", index=False)
