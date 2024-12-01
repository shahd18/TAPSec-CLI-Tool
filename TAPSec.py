# Imports
import joblib
import pandas as pd
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import typer
from art import text2art
import logging
import sys
import threading
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import numpy as np
from colorama import Fore, Style, init
import textwrap
import torch
import time
import os
from tabulate import tabulate
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# Suppress unnecessary logs from specific libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Initialize Typer app for command-line interface
app = typer.Typer()

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Load stopwords and spaCy model for text processing
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

# Constants and Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFIDF_VECTORIZER_PATH = os.path.join(BASE_DIR, 'Classification-Model', 'tfidf_vectorizer.pkl')
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, 'Classification-Model', 'xgboost_best_model.joblib')
CVE_MODEL = os.path.join(BASE_DIR ,'CVE-Model')
MITRE_MODEL_DIR = os.path.join(BASE_DIR, 'MITRE-Model')  
KMEAN_MODEL = os.path.join(BASE_DIR, 'KMeans-Model', 'kmeans_model_FRI.pkl')
CVE_DATASET_PATH = os.path.join(BASE_DIR, 'Dataset', 'Processed_CVE_withSpace.csv')
MITRE_DATASET_PATH = os.path.join(BASE_DIR, 'Dataset', 'Processed_mitre_recommidtaion.csv') 
KEYWORDS_DATASET_PATH = os.path.join(BASE_DIR, 'Dataset', 'combinedRandomly_topic_names_withTarget.csv')  
TOOL_DESIGN_PATH = os.path.join(BASE_DIR, 'Design' ,'tooldesign1.txt')

# Thresholds for classification and similarity
CLASSIFICATION_THRESHOLD = 0.45
SIMILARITY_THRESHOLD = 0.7


# Utility Functions

def preprocess_text(text):
    """
    Preprocess the input text by removing non-alphabetic characters,
    converting to lowercase, tokenizing, removing stopwords, and lemmatizing.

    :param text: The input text to preprocess.
    :return: The processed text as a single string.
    """
    if isinstance(text, float):  # Handle NaN or float cases
        text = ''
    text = re.sub(r'\W', ' ', text.lower())  # Remove special characters
    words = word_tokenize(text)  # Tokenize words
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    doc = nlp(" ".join(words))  # Lemmatize words
    return ' '.join(token.lemma_ for token in doc)


# Display ASCII Art
def display_ascii_art():
    ascii_art = text2art("          TAPsec")
    print(ascii_art)

# Read the content of tooldesign.txt for help messages
try:
    with open(TOOL_DESIGN_PATH , 'r') as file:
        tool_design_content = file.read()
except FileNotFoundError:
    tool_design_content = "Design file not found."


def wrap_and_pad_text(text, width):
    """Wrap and pad text for display."""
    wrapped_text = textwrap.fill(text, width=width)
    return wrapped_text

def split_and_wrap_mitigation(mitigation, width):
    """Split and wrap mitigation text."""
    mitigation_parts = mitigation.split('.')
    wrapped_mitigation = ''
    for part in mitigation_parts:
        part = part.strip()  # Remove leading/trailing whitespace
        if part:
            wrapped_mitigation += wrap_and_pad_text(part, width=width) + '\n'
    return wrapped_mitigation.strip()

# Adjust text wrapping to apply color properly
def wrap_and_color_text(text, width, color):
    from textwrap import wrap
    wrapped_lines = wrap(text, width=width)
    return "\n".join([color + line + Style.RESET_ALL for line in wrapped_lines])

def loading_animation(message):
    """Display a loading animation."""
    while not stop_animation_event.is_set():
        for cursor in '|/-\\':
            if stop_animation_event.is_set():  # Check again before printing
                break
            print(f'\r{message} {cursor}', end='', flush=True)
            time.sleep(0.1)
    print('\r' + Fore.GREEN + "\n\nDone ._. *_*\n" + Style.RESET_ALL)  # Ensure proper cleanup of the last message



# Main Processing Functions

def classify_input(rule, app_name):
    """
    Classify the combined input of rule and app name as malicious or innocent.

    :param rule: The rule input from the user.
    :param app_name: The application name input from the user.
    :return: A tuple indicating if the input is malicious and the probability of being malicious.
    """
    user_input = f"{rule} {app_name}"
    processed_input = preprocess_text(user_input)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    classification_model = joblib.load(CLASSIFIER_MODEL_PATH)
    input_tfidf = tfidf_vectorizer.transform([processed_input])
    malicious_prob = classification_model.predict_proba(input_tfidf)[0][1]
    is_malicious = malicious_prob >= CLASSIFICATION_THRESHOLD
    return is_malicious, malicious_prob

def map_user_input_to_topic(rule, app_name, keywords_df):
    """
    Map the user input to the best matching topic based on keywords.

    :param rule: The rule input from the user.
    :param app_name: The application name input from the user.
    :param keywords_df: DataFrame containing keywords and their mappings.
    :return: A dictionary with the best matched keyword, generated topic name, and target value.
    """
    combined_input = f"{rule} {app_name}"
    processed_combined_input = preprocess_text(combined_input)
    model = SentenceTransformer('roberta-large')
    keywords = keywords_df['Keywords'].tolist()
    keywords_embeddings = model.encode(keywords, convert_to_tensor=True)
    combined_input_embedding = model.encode(processed_combined_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(combined_input_embedding, keywords_embeddings).squeeze()
    best_match_idx = cosine_scores.argmax().item()
    best_matched_keyword = keywords_df.loc[best_match_idx, 'Keywords']
    generated_topic_name = keywords_df.loc[best_match_idx, 'Generated Topic Name']
    target_value = keywords_df.loc[best_match_idx, 'Target']
    return {
        'Best Matched Keyword': best_matched_keyword,
        'Generated Topic Name': generated_topic_name,
        'Target': target_value
    }

# Function to compute embeddings
def compute_embeddings(texts, model, tokenizer, max_length=64):
    """Compute embeddings for a list of texts."""
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length').to('cpu')
        with torch.no_grad():
            embedding = model(**inputs).pooler_output  # Extract the pooler output (CLS token representation)
        embeddings.append(embedding.squeeze().cpu().numpy())
    return np.array(embeddings)

def find_best_cve_match(rule, app_name, mapped_topic, mapped_topic_name, cve_df, kmeans_model, bert_model, tokenizer, similarity_threshold=0.7):
    """
    Find the best matching CVE for the given user input.

    :param rule: The rule input from the user.
    :param app_name: The application name input from the user.
    :param mapped_topic: The best matched keyword from user input.
    :param mapped_topic_name: The generated topic name from user input.
    :param cve_df: DataFrame containing CVE data.
    :param kmeans_model: Pre-trained KMeans model for clustering.
    :param bert_model: Pre-trained BERT model for embeddings.
    :param tokenizer: Tokenizer for processing CVE texts.
    :param similarity_threshold: Threshold for cosine similarity to consider a valid match.
    :return: The best matching CVE details and a flag indicating if it is a valid match.
    """
     
    user_input = preprocess_text(f"{rule} {app_name} {mapped_topic} {mapped_topic_name}")
    user_embedding = compute_embeddings([user_input], bert_model, tokenizer).squeeze()
    cve_texts = cve_df['Processed_Text'].tolist()  
    cve_embeddings = compute_embeddings(cve_texts, bert_model, tokenizer)
    predicted_cluster = kmeans_model.predict(user_embedding.reshape(1, -1))[0]
    similarities = cosine_similarity(user_embedding.reshape(1, -1), cve_embeddings).flatten()
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]


    true_cve_cluster = kmeans_model.predict(cve_embeddings[best_match_idx].reshape(1, -1))[0]
    cve_name = cve_df.iloc[best_match_idx]['Name'] if 'Name' in cve_df.columns else 'N/A'
    cve_text = cve_df.iloc[best_match_idx]['Processed_Text']
    cve_full_text = cve_df.iloc[best_match_idx]['Text']

    is_valid_match = predicted_cluster == true_cve_cluster and best_similarity >= similarity_threshold

    return cve_name, cve_text, cve_full_text, best_similarity, is_valid_match


# MITRE Processing
# Pre-select relevant MITRE techniques using TF-IDF
def preselect_mitre(cve_texts, mitre_texts, top_n=5):
    """
    Use TF-IDF to preselect the most relevant MITRE techniques for each CVE description.
    
    :param cve_texts: List of CVE descriptions.
    :param mitre_texts: List of MITRE technique descriptions.
    :param top_n: Number of top techniques to preselect per CVE.
    :return: List of indices for preselected techniques for each CVE.
    """
    vectorizer = BertTokenizer.from_pretrained(MITRE_MODEL_DIR)
    tfidf_cve = vectorizer.fit_transform(cve_texts)
    tfidf_mitre = vectorizer.transform(mitre_texts)

    preselected_mitre_indices = []
    for i in range(tfidf_cve.shape[0]):
        similarities = cosine_similarity(tfidf_cve[i], tfidf_mitre).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        preselected_mitre_indices.append(top_indices)

    return preselected_mitre_indices


# Dataset for CVE-MITRE pairs
class CVEMITREDataset(Dataset):
    def __init__(self, cve_texts, mitre_texts, preselected_indices, tokenizer, max_length=128):
        self.cve_texts = cve_texts
        self.mitre_texts = mitre_texts
        self.preselected_indices = preselected_indices
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_cve = [
            self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
            for text in cve_texts
        ]
        self.tokenized_mitre = [
            self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True, padding='max_length')
            for text in mitre_texts
        ]

    def __len__(self):
        return sum(len(indices) for indices in self.preselected_indices)

    def __getitem__(self, idx):
        cumulative_count = 0
        for cve_idx, indices in enumerate(self.preselected_indices):
            if cumulative_count + len(indices) > idx:
                mitre_local_idx = idx - cumulative_count
                mitre_idx = indices[mitre_local_idx]
                break
            cumulative_count += len(indices)

        inputs_cve = self.tokenized_cve[cve_idx]
        inputs_mitre = self.tokenized_mitre[mitre_idx]

        return {
            'input_ids_cve': inputs_cve['input_ids'].squeeze(0),
            'attention_mask_cve': inputs_cve['attention_mask'].squeeze(0),
            'input_ids_mitre': inputs_mitre['input_ids'].squeeze(0),
            'attention_mask_mitre': inputs_mitre['attention_mask'].squeeze(0),
            'cve_idx': cve_idx,
            'mitre_idx': mitre_idx
        }
    
def process_selected_cve_with_mitre(selected_cve_text, mitre_df, tokenizer_mitre, bert_model_mitre, threshold=0.6, top_n=5):
    """
    Match the selected CVE description with MITRE techniques based on embeddings.

    :param selected_cve_text: Text of the selected CVE to be matched with MITRE.
    :param mitre_df: DataFrame containing MITRE techniques and metadata.
    :param tokenizer_mitre: Tokenizer for processing MITRE texts.
    :param bert_model_mitre: Pre-trained BERT model for embeddings.
    :param threshold: Similarity threshold for a valid match.
    :param top_n: Number of top techniques to consider.
    :return: List of matched techniques, tactics, and mitigations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model_mitre.to(device)
    bert_model_mitre.eval()
    
    # Tokenize and encode the selected CVE
    selected_cve_inputs = tokenizer_mitre(selected_cve_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
    with torch.no_grad():
        selected_cve_embedding = bert_model_mitre(**selected_cve_inputs).pooler_output.cpu().numpy()
    
    # Process MITRE techniques
    mitre_texts = mitre_df['Processed_Technique'].tolist()
    mitre_embeddings = []
    for mitre_text in mitre_texts:
        mitre_inputs = tokenizer_mitre(mitre_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(device)
        with torch.no_grad():
            mitre_embedding = bert_model_mitre(**mitre_inputs).pooler_output.cpu().numpy()
        mitre_embeddings.append(mitre_embedding)

    mitre_embeddings = np.vstack(mitre_embeddings)

    # Compute cosine similarity
    similarities = cosine_similarity(selected_cve_embedding, mitre_embeddings).flatten()
    
    # Get top N matches based on similarity
    top_indices = similarities.argsort()[-top_n:][::-1]
    matched_results = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            matched_results.append({
                'Matched Techniques': mitre_df.iloc[idx]['Processed_Technique'],
                'Tactic': mitre_df.iloc[idx]['Tactic'],
                'Mitigation': mitre_df.iloc[idx]['mitigation'],
                'Similarity': similarities[idx]
            })

    return matched_results

# Output Functions

def print_output_tables(rule, best_cve_name, best_cve_original_text, mitre_results, rule_col_width=50, cve_name_col_width=20, cve_text_col_width=60, technique_col_width=25, tactic_col_width=15, mitigation_col_width=70, border_length=140, table_style="grid"):
    """
    Print results in a formatted table, including the rule, best CVE match, and MITRE matches.

    :param rule: The input rule provided by the user.
    :param best_cve_name: The name of the best-matching CVE.
    :param best_cve_original_text: The original text of the best-matching CVE.
    :param mitre_results: List of dictionaries with matched MITRE techniques, tactics, and mitigations.
    :param rule_col_width: Width of the Rule column.
    :param cve_name_col_width: Width of the CVE Name column.
    :param cve_text_col_width: Width of the CVE Text column.
    :param technique_col_width: Width of the MITRE Techniques column.
    :param tactic_col_width: Width of the Tactic column.
    :param mitigation_col_width: Width of the Mitigation column.
    :param border_length: Length of the table border.
    :param table_style: Style of the table (e.g., 'grid').
    """
    wrapped_rule = wrap_and_pad_text(rule, width=rule_col_width)
    wrapped_cve_name = wrap_and_pad_text(best_cve_name, width=cve_name_col_width)
    wrapped_cve_text = wrap_and_pad_text(best_cve_original_text, width=cve_text_col_width)

    print("\n")
    print(tabulate(pd.DataFrame({'Rule': [wrapped_rule]}), headers='keys', tablefmt=table_style, showindex=False))
    print("\n")
    print(tabulate(pd.DataFrame({'CVE Name': [Fore.RED + wrapped_cve_name + Style.RESET_ALL], 'Original Text': [wrapped_cve_text]}), headers='keys', tablefmt=table_style, showindex=False))
    print("\n")

    if mitre_results:
        mitre_table = []
        for result in mitre_results:
            wrapped_tactic = wrap_and_color_text(result['Tactic'], tactic_col_width, Fore.RED)
            wrapped_techniques = wrap_and_color_text(result['Matched Techniques'], technique_col_width, Fore.CYAN)
            wrapped_mitigation = split_and_wrap_mitigation(result['Mitigation'], mitigation_col_width)
            mitre_table.append({'MITRE Tactic': wrapped_tactic, 'MITRE Techniques': wrapped_techniques, 'Mitigation': wrapped_mitigation})

        print(tabulate(pd.DataFrame(mitre_table), headers='keys', tablefmt=table_style, showindex=False))
    else:
        print(Fore.YELLOW + "No MITRE matches found." + Style.RESET_ALL)

# Typer Command: Main Function

@app.command(help=tool_design_content)
def main(rule: str = typer.Option(..., "-r", "--rule", help="Rule input"), app_name: str = typer.Option(..., "-a", "--app-name", help="App name")):
    """
    Main entry point for the Typer command-line application.

    :param rule: The rule input provided by the user.
    :param app_name: The name of the application provided by the user.
    """
    try:
        # Validate rule input
        if not all(part.isalpha() for part in rule.split()):
            print(Fore.RED + "Error: The rule input must contain only alphabetic characters and spaces (no numbers or special characters)." + Style.RESET_ALL)
            sys.exit(1)
        
        # Validate app name input
        if not all(part.isalpha() for part in app_name.split()):
            print(Fore.RED + "Error: The app name must contain only alphabetic characters and spaces (no numbers or special characters)." + Style.RESET_ALL)
            sys.exit(1)

        print(Fore.CYAN + "Classifying input..." + Style.RESET_ALL)
        is_malicious, malicious_prob = classify_input(rule, app_name)

        if not is_malicious:
            print(Fore.GREEN + "The Rule is classified as Innocent. Exiting the tool." + Style.RESET_ALL)
            sys.exit(0)

        print(Fore.RED + f"The Rule is classified as Malicious (Probability: {malicious_prob:.2f})." + Style.RESET_ALL)

        display_ascii_art()
        
        print(tool_design_content)
        
        # Start loading animation in a separate thread
        global stop_animation_event
        stop_animation_event = threading.Event()
        animation_thread = threading.Thread(target=loading_animation, args=("Loading... ",))
        animation_thread.start()

        # Load datasets
        cve_df = pd.read_csv(CVE_DATASET_PATH)
        mitre_df = pd.read_csv(MITRE_DATASET_PATH)
        keywords_df = pd.read_csv(KEYWORDS_DATASET_PATH)

        # Load tokenizers and models
        tokenizer_cve = BertTokenizer.from_pretrained(CVE_MODEL)
        model_cve = BertModel.from_pretrained(CVE_MODEL)
        tokenizer_mitre = BertTokenizer.from_pretrained(MITRE_MODEL_DIR)
        bert_model_mitre = BertModel.from_pretrained(MITRE_MODEL_DIR)

        # Load KMeans model
        kmeans_model =  joblib.load(KMEAN_MODEL)

        # Find the best topic match
        topic_mapping = map_user_input_to_topic(rule, app_name, keywords_df)

        best_matched_keyword = topic_mapping['Best Matched Keyword']
        generated_topic_name = topic_mapping['Generated Topic Name']
        target_value = topic_mapping['Target']

        # Find the best CVE match
        best_cve_name, best_cve_text, best_cve_original_text, best_similarity, is_valid_match = find_best_cve_match(
            rule, app_name, best_matched_keyword, generated_topic_name, cve_df, kmeans_model, model_cve, tokenizer_cve, similarity_threshold=0.6
        )

        if not is_valid_match:
            print(Fore.YELLOW + "\nNo valid CVE match found based on cluster similarity :( \n" + Style.RESET_ALL)
            stop_animation_event.set()  # Stop animation on error
            sys.exit(0)

        selected_cve_text = best_cve_text
        
        # Process MITRE matching
        mitre_results = process_selected_cve_with_mitre(
            selected_cve_text=selected_cve_text,
            mitre_df=mitre_df,
            tokenizer_mitre=tokenizer_mitre,
            bert_model_mitre=bert_model_mitre,
            threshold=0.6,
            top_n=5
        )

      
        # Print the results in structured tables
        print_output_tables(
            rule=rule,
            best_cve_name=best_cve_name,
            best_cve_original_text=best_cve_original_text,
            mitre_results=mitre_results,
            rule_col_width=50,
            cve_name_col_width=25,
            cve_text_col_width=70,
            technique_col_width=30,
            tactic_col_width=20,
            mitigation_col_width=70,
            border_length=160,
            table_style="grid"
        )

        # Stop loading animation
        stop_animation_event.set()
        animation_thread.join()

        # Save results to CSV
        output_df = pd.DataFrame({
            'Rule': [rule],
            'App Name': [app_name],
            'Best Matched Keyword': [best_matched_keyword],
            'Generated Topic Name': [generated_topic_name],
            'Target': [target_value],
            'CVE Name': [best_cve_name],
            'CVE Processed Text': [best_cve_text],
            'CVE Original Text': [best_cve_original_text],
            'Similarity Score': [best_similarity],
            'Matched MITRE Techniques': [mitre_results[0]['Matched Techniques'] if mitre_results else 'N/A'],
            'Tactic': [mitre_results[0]['Tactic'] if mitre_results else 'N/A'],
            'Mitigation': [mitre_results[0]['Mitigation'] if mitre_results else 'N/A']
        })
        output_file_path = f"D:\\combined4_processed_{app_name}_output.csv"
        output_df.to_csv(output_file_path, index=False)


    except Exception as e:
        stop_animation_event.set()  # Stop animation on error
        print(Fore.RED + f"An error occurred: {str(e)}" + Style.RESET_ALL)
        sys.exit(1)

    finally:
        animation_thread.join() 


if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred: {str(e)}" + Style.RESET_ALL)
        sys.exit(1)