import time
from gensim.models import Word2Vec

# Train Skip-gram model with training time
def train_skipgram_model(sentences):
    start_time = time.time()
    skipgram_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=5, sg=1, workers=-1)
    skipgram_time = time.time() - start_time
    return skipgram_model, skipgram_time