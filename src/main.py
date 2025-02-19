import pandas as pd
import numpy as np
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

sentences = dl.load_data("data/processed_discharge.csv")

skipgram_model, skipgram_time = skipgram.train_skipgram_model(sentences)
print(skipgram_time)

embeddings_sppmi_svd, sppmi_svd_time = SPPMI.train_sppmi_svd_model(sentences)
print(sppmi_svd_time)

glove_model, glove_time = GloVe.train_glove_model(sentences)
print(glove_model)
print(glove_time)

# # Train SPPMI-SVD model with training time
# def train_sppmi_svd_model(sentences):
#     start_time = time.time()
#     vocab = Counter(itertools.chain(*sentences.tolist()))
#     word_to_index = {word: i for i, word in enumerate(vocab.keys())}
#     index_to_word = {i: word for word, i in word_to_index.items()}
#     cooccurrence = defaultdict(Counter)

#     for sentence in sentences:
#         indices = [word_to_index[word] for word in sentence if word in word_to_index]
#         for i, idx in enumerate(indices):
#             for j in range(max(0, i - 5), min(len(indices), i + 5 + 1)):
#                 if i != j:
#                     cooccurrence[idx][indices[j]] += 1

#     # Convert to Dense Matrix
#     vocab_size = len(vocab)
#     cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
#     for word_idx, context_counts in cooccurrence.items():
#         for context_idx, count in context_counts.items():
#             cooccurrence_matrix[word_idx, context_idx] = count

#     # Compute PPMI Efficiently
#     word_sum = np.sum(cooccurrence_matrix, axis=1, keepdims=True)  # Row sum
#     total_sum = np.sum(word_sum)  # Total sum
#     ppmi_matrix = np.maximum(np.log((cooccurrence_matrix * total_sum) / (word_sum * word_sum.T)), 0)

#     # Apply SVD for embeddings
#     svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
#     embeddings_sppmi_svd = svd.fit_transform(ppmi_matrix)
#     sppmi_svd_time = time.time() - start_time
#     return embeddings_sppmi_svd, sppmi_svd_time


# # # Load data
# # file_path = "data/processed_discharge.csv"
# # df = pd.read_csv(file_path)

# # # Preprocess text
# # def preprocess_text(text):
# #     return gensim.utils.simple_preprocess(str(text))

# # df["tokenized_text"] = df["text"].dropna().apply(preprocess_text)

# # # Measure training time for Skip-gram
# # start_time = time.time()
# # skipgram_model = Word2Vec(
# #     sentences=df["tokenized_text"],
# #     vector_size=100,
# #     window=5,
# #     min_count=5,
# #     sg=1,
# #     workers=-1  # Use all CPU cores
# # )
# # skipgram_time = time.time() - start_time
# # skipgram_model.save("skipgram.model")

# # Measure training time for SPPMI-SVD
# start_time = time.time()
# vocab = Counter(itertools.chain(*df["tokenized_text"].tolist()))
# word_to_index = {word: i for i, word in enumerate(vocab.keys())}
# index_to_word = {i: word for word, i in word_to_index.items()}
# cooccurrence = defaultdict(Counter)

# for sentence in df["tokenized_text"]:
#     indices = [word_to_index[word] for word in sentence if word in word_to_index]
#     for i, idx in enumerate(indices):
#         for j in range(max(0, i - 5), min(len(indices), i + 5 + 1)):
#             if i != j:
#                 cooccurrence[idx][indices[j]] += 1

# # Convert to Dense Matrix
# vocab_size = len(vocab)
# cooccurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
# for word_idx, context_counts in cooccurrence.items():
#     for context_idx, count in context_counts.items():
#         cooccurrence_matrix[word_idx, context_idx] = count

# # Compute PPMI Efficiently
# word_sum = np.sum(cooccurrence_matrix, axis=1, keepdims=True)  # Row sum
# total_sum = np.sum(word_sum)  # Total sum
# ppmi_matrix = np.maximum(np.log((cooccurrence_matrix * total_sum) / (word_sum * word_sum.T)), 0)

# # Apply SVD for embeddings
# svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
# embeddings_sppmi_svd = svd.fit_transform(ppmi_matrix)
# sppmi_svd_time = time.time() - start_time
# np.save("sppmi_svd_embeddings.npy", embeddings_sppmi_svd)

# # Measure training time for FastText
# start_time = time.time()
# fasttext_model = FastText(
#     sentences=df["tokenized_text"],
#     vector_size=100,
#     window=5,
#     min_count=5,
#     sg=1,
#     workers=-1
# )
# fasttext_time = time.time() - start_time
# fasttext_model.save("fasttext.model")

# # Store training times
# training_times = pd.DataFrame(
#     {"Model": ["Skip-gram", "SPPMI-SVD", "FastText"], "Training Time (seconds)": [skipgram_time, sppmi_svd_time, fasttext_time]}
# )
# print(training_times)

# # Visualization with PCA & t-SNE for Key Medical Terms
# key_terms = ["diabetes", "hypertension", "insulin", "surgery", "cbc", "antibiotic"]

# # Function to extract valid word vectors safely
# def get_word_vectors(model, words):
#     valid_words = [word for word in words if word in model.wv]
#     vectors = np.array([model.wv[word] for word in valid_words])
#     return valid_words, vectors

# # Extract vectors
# skipgram_words, skipgram_vectors = get_word_vectors(skipgram_model, key_terms)
# fasttext_words, fasttext_vectors = get_word_vectors(fasttext_model, key_terms)

# # PCA transformation
# pca = PCA(n_components=2)
# skipgram_pca = pca.fit_transform(skipgram_vectors)
# fasttext_pca = pca.fit_transform(fasttext_vectors)

# # Plot PCA results
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# axes[0].scatter(skipgram_pca[:, 0], skipgram_pca[:, 1], color="blue")
# for i, word in enumerate(skipgram_words):
#     axes[0].annotate(word, (skipgram_pca[i, 0], skipgram_pca[i, 1]))
# axes[0].set_title("PCA of Skip-gram Embeddings")

# axes[1].scatter(fasttext_pca[:, 0], fasttext_pca[:, 1], color="red")
# for i, word in enumerate(fasttext_words):
#     axes[1].annotate(word, (fasttext_pca[i, 0], fasttext_pca[i, 1]))
# axes[1].set_title("PCA of FastText Embeddings")

# plt.show()

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
