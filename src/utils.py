import numpy as np
from collections import Counter, defaultdict

def build_cooccurrence_matrix(sentences, window_size=5):
    """Builds a word co-occurrence matrix from sentences."""
    vocab = set(word for sentence in sentences for word in sentence)
    print(len(vocab))
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    
    cooccurrence_matrix = defaultdict(float)
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

def get_keywords():
    keywords = {
        "Diseases": [
            "pneumonia", "sepsis", "hypertension", "diabetes", "heart", "stroke", 
            "asthma", "COPD", "myocardial", "infarction", "anemia", "cancer", 
            "liver", "disease", "kidney", "failure", "leukemia", "HIV",
            "tuberculosis", "alzheimer", "arthritis", "depression", "schizophrenia"
        ],
        
        "Medications": [
            "aspirin", "ibuprofen", "acetaminophen", "warfarin", "heparin",
            "metformin", "insulin", "amlodipine", "lisinopril", "atorvastatin",
            "morphine", "furosemide", "omeprazole", "prednisone", "azithromycin",
            "ciprofloxacin", "clopidogrel", "simvastatin", "albuterol", "diazepam"
        ],
        
        "Procedures": [
            "intubation", "dialysis", "colonoscopy", "endoscopy", "bypass", "surgery",
            "angioplasty", "biopsy", "appendectomy", "cataract", "surgery", "hysterectomy",
            "lumbar", "puncture", "MRI", "scan", "CT", "scan", "ultrasound", "ventilation",
            "thoracentesis", "pacemaker", "insertion", "stent", "placement", "tracheostomy"
        ],
        
        "Laboratory_tests": [
            "complete", "blood", "count", "culture", "electrolyte", "glucose", 
            "test", "lipid", "panel", "liver", "function", "kidney", "urinalysis", 
            "coagulation", "troponin", "C-reactive", "protein", "bilirubin", "gas", 
            "analysis", "hemoglobin", "A1C", "thyroid", "amylase", "lactic", "acid"
        ]
    }
    reversed_keywords = {word: category for category, words in keywords.items() for word in words}
    return reversed_keywords
