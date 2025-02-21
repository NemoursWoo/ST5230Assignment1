library(dplyr)
library(text2vec)
library(NLP)
library(tm)
library(stringr)
library(stopwords)
library(Rtsne)
library(lsa)

data <- read.csv("Preprocessing_discharge.csv")
# Tokenize the sentences
tokens <- word_tokenizer(data$text)

# Create an iterator over the tokens
it <- itoken(tokens, progressbar = FALSE)

# Build the vocabulary
vocab <- create_vocabulary(it)

# Prune the vocabulary (optional: remove infrequent or frequent terms)
vocab <- prune_vocabulary(vocab, term_count_min = 1)

#start measuring time
start_time <- Sys.time()

# Create a term-co-occurrence matrix (TCM)
tcm <- create_tcm(it, vectorizer = vocab_vectorizer(vocab), skip_grams_window = 10)



# Define the GloVe model
glove <- GlobalVectors$new(rank = 50, x_max = 100)  # rank = embedding dimensions

# Fit the GloVe model
word_vectors <- glove$fit_transform(tcm, n_iter = 20, convergence_tol = 0.05)

# Measure the training time
training_time_glove <- Sys.time() - start_time

# Print the training time
print(paste("GloVe Training Time:", training_time_glove))

# Combine word and context embeddings (optional)
word_vectors <- word_vectors + t(glove$components)


library(dplyr)

#build keywords list
keywords <- list(
  Diseases = c("pneumonia", "sepsis", "hypertension", "diabetes", "heart", "failure",
               "stroke", "asthma", "COPD", "myocardial", "infarction", "anemia",
               "cancer", "liver", "disease", "kidney", "failure", "leukemia", "HIV",
               "tuberculosis", "alzheimer", "arthritis", "depression", "schizophrenia"),
  
  Medications = c("aspirin", "ibuprofen", "acetaminophen", "warfarin", "heparin",
                  "metformin", "insulin", "amlodipine", "lisinopril", "atorvastatin",
                  "morphine", "furosemide", "omeprazole", "prednisone", "azithromycin",
                  "ciprofloxacin", "clopidogrel", "simvastatin", "albuterol", "diazepam"),
  
  Procedures = c("intubation", "dialysis", "colonoscopy", "endoscopy", "bypass", "surgery",
                 "angioplasty", "biopsy", "appendectomy", "cataract", "surgery", "hysterectomy",
                 "lumbar", "puncture", "MRI", "scan", "CT", "scan", "ultrasound", "ventilation",
                 "thoracentesis", "pacemaker", "insertion", "stent", "placement", "tracheostomy"),
  
  Laboratory_tests = c("complete", "blood", "count", "blood", "culture", "electrolyte", "panel",
                       "glucose", "test", "lipid", "panel", "liver", "function", "test",
                       "kidney", "function", "test", "urinalysis", "coagulation", "test",
                       "troponin", "test", "C-reactive", "protein", "bilirubin", "test",
                       "blood", "gas", "analysis", "hemoglobin", "A1C", "thyroid", "function", "test",
                       "amylase", "test", "lactic", "acid", "test", "procalcitonin", "test")
)

all_keywords <- unlist(keywords)
#select all keywords from vocab
filtered_vocab <- vocab %>%
  filter(term %in% all_keywords)

print(filtered_vocab)

key_terms<-filtered_vocab
key_terms_embedding <- word_vectors[key_terms$term, ]


library(ggplot2)
set.seed(123)  # Set the seed for reproducibility

# Perform PCA to reduce dimensions to 2D
pca <- prcomp(key_terms_embedding, center = TRUE, scale. = TRUE)
key_terms_pca <- data.frame(pca$x[, 1:2])
key_terms_pca$word <- rownames(key_terms_embedding)


# Plot the embeddings
ggplot(key_terms_pca, aes(x = PC1, y = PC2, label = word)) +
  geom_point() +
  geom_text(aes(label = word), hjust = 0, vjust = 1, size = 3) +
  theme_minimal() +
  labs(title = "key terms Word Embeddings Visualization", x = "PC1", y = "PC2")


# Perform t-SNE to reduce dimensions to 2D

tsne <- Rtsne(key_terms_embedding, dims = 2, pca = TRUE, perplexity = 10, check_duplicates = FALSE)

# Extract t-SNE results
key_terms_tsne <- data.frame(tsne$Y)
key_terms_tsne$word <- rownames(key_terms_embedding)

# Plot the t-SNE visualization

ggplot(key_terms_tsne, aes(x = X1, y = X2, label = word)) +
  geom_point() +
  geom_text(aes(label = word), hjust = 0, vjust = 1, size = 3) +
  theme_minimal() +
  labs(title = "key Terms t-SNE Visualization", x = "t-SNE 1", y = "t-SNE 2")


# Function to calculate cosine similarity
cosine_similarity <- function(vec1, vec2) {
  cosine(vec1, vec2)
}


# Calculate cosine similarity
similarity_results <- data.frame(term1 = character(0), term2 = character(0), similarity = numeric(0))

for (i in 1:85) {
  for (j in (i+1):86) {
    term1 <- key_terms[i,]
    term2 <- key_terms[j,]
    
    embedding_word1 <- key_terms_embedding[term1$term, ]
    embedding_word2 <- key_terms_embedding[term2$term, ]
    
    # cosine similarity
    similarity <- cosine_similarity(embedding_word1, embedding_word2)
    
    similarity_results <- rbind(similarity_results, data.frame(term1 = term1$term, term2 = term2$term, similarity = similarity))
  }
}

# top_10_similarities
top_10_similarities <- similarity_results %>%
  arrange(desc(similarity)) %>%
  head(10)

print(top_10_similarities)


