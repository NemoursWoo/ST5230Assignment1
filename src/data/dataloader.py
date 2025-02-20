# PhysioNet's MIMIC-IV dataset is a large, freely available database comprising deidentified health-related data associated with over 280,000 admissions to critical care units
# https://physionet.org/content/mimic-iv-note/2.2/note/
import os
import pandas as pd
# import gensim

def load_data(use_preprocessed_data_ipynb=False):
    # Get absolute path to the file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if use_preprocessed_data_ipynb:
        input_file_path = os.path.join(base_dir, "preprocessed_discharge.csv")
        output_file_path = os.path.join(base_dir, "preprocessed_discharge_tokenized.csv")

        # If the tokenized file already exists, load and return it
        if os.path.exists(output_file_path):
            print("Loading existing tokenized preprocessed text file...")
            df = pd.read_csv(output_file_path)
            return df["tokenized_text"].apply(eval)  # Convert string representation of list back to list
        
        print("Tokenizing preprocessed text data...")
        df = pd.read_csv(input_file_path)

        # Preprocess text
        def preprocess_text(text):
            tokens = gensim.utils.simple_preprocess(str(text))
            return [word for word in tokens if word.lower()]
        
        df["tokenized_text"] = df["text"].dropna().apply(preprocess_text)
        
        # Save the tokenized text as a new CSV file in the same directory
        df[["tokenized_text"]].to_csv(output_file_path, index=False)

        return df["tokenized_text"]
    else:
        input_file_path = os.path.join(base_dir, "processed_discharge.csv")
        output_file_path = os.path.join(base_dir, "processed_discharge_tokenized.csv")

        # If the tokenized file already exists, load and return it
        if os.path.exists(output_file_path):
            print("Loading existing tokenized text file...")
            df = pd.read_csv(output_file_path)
            return df["tokenized_text"].apply(eval)  # Convert string representation of list back to list
        
        print("Processing and tokenizing text data...")
        df = pd.read_csv(input_file_path)

        # List of words to remove
        remove_words = {"name", "unit", "no", "admission", "date", "discharge", "of", "birth", "sex", "service", "or", "and", "known", "with", "this", "is", "attending"}
        
        # Preprocess text
        def preprocess_text(text):
            tokens = gensim.utils.simple_preprocess(str(text))
            return [word for word in tokens if word.lower() not in remove_words]

        df["tokenized_text"] = df["text"].dropna().apply(preprocess_text)
        
        # Save the tokenized text as a new CSV file in the same directory
        df[["tokenized_text"]].to_csv(output_file_path, index=False)

        return df["tokenized_text"]
