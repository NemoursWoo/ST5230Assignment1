# PhysioNet's MIMIC-IV dataset is a large, freely available database comprising deidentified health-related data associated with over 280,000 admissions to critical care units
# https://physionet.org/content/mimic-iv-note/2.2/note/
import os
import pandas as pd
import gensim

# Load proprecessed data and preprocess text
def load_data(file_path):
    # Get absolute path to the file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "processed_discharge.csv")
    df = pd.read_csv(file_path)
    
    # List of words to remove
    remove_words = {"name", "unit", "no", "admission", "date", "discharge"}

    # Preprocess text
    def preprocess_text(text):
        tokens = gensim.utils.simple_preprocess(str(text))
        return [word for word in tokens if word.lower() not in remove_words]
    
    df["tokenized_text"] = df["text"].dropna().apply(preprocess_text)
    return df["tokenized_text"]
