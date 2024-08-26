import pandas as pd
import re
import stanza
import xml.etree.ElementTree as ET
from transformers import BertTokenizer, AutoModelForTokenClassification
import torch
from nltk.tokenize import sent_tokenize
import os
import numpy as np
def load_unique_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        unique_names = [line.strip() for line in file.readlines()]
    return unique_names
import re
import os
import re
import xml.etree.ElementTree as ET
def find_name_occurrences(names, clinical_letters):
    name_positions = []
    for name in names:
        pattern = r'\b{}\b'.format(re.escape(name))
        matches = [(name, m.start(), m.end()) for m in re.finditer(pattern, clinical_letters, re.IGNORECASE)]
        if matches:
            name_positions.extend(matches)
    return name_positions
# Define the file paths
unique_names_file_path = 'unique_names.txt'  # The file generated previously with unique names
clinical_letters_file_path = 'dataset/training-PHI-Gold-Set1/220-03.xml'  # Path to the clinical letters XML file

# Load unique names
unique_names = load_unique_names(unique_names_file_path)

# Load clinical letters from XML file
def load_clinical_letters_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    text = root.find(".//TEXT").text
    return text

clinical_letters = load_clinical_letters_from_xml(clinical_letters_file_path)

# Find occurrences and positions of unique names
name_occurrences = find_name_occurrences(unique_names, clinical_letters)

# Print the results
print("Name occurrences in clinical letters:")
for name, start, end in name_occurrences:
    print(f"Name: {name}, Start: {start}, End: {end}")

# Optionally, write occurrences to a file
with open('name_occurrences.txt', 'w', encoding='utf-8') as f:
    for name, start, end in name_occurrences:
        f.write(f"{name}\t{start}\t{end}\n")

print("Name occurrences have been written to 'name_occurrences.txt'.")

# Download and initialize the English model for Stanza
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,ner')

# Load the first column from a CSV file
def load_first_column_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    names = df.iloc[:, 0].dropna().tolist()
    return names


# Use Stanza for Named Entity Recognition
def detect_names_with_stanza(clinical_letters):
    doc = nlp(clinical_letters)
    detected_entities = [ent.text for ent in doc.ents if ent.type == 'PERSON']
    return detected_entities

# Verify if the detected names are in the dataset
def verify_names(detected_entities, names):
    verified_names = set()
    name_parts = set(name.lower() for name in names)
    for entity in detected_entities:
        parts = entity.split()
        for part in parts:
            if part.lower() in name_parts:
                verified_names.add(part)
    return list(verified_names)

# Find occurrences and their positions of each name in the clinical letters
# def find_name_occurrences(names, clinical_letters):
#     name_positions = []
#     for name in names:
#         pattern = r'\b{}\b'.format(re.escape(name))
#         matches = [(name, m.start(), m.end()) for m in re.finditer(pattern, clinical_letters, re.IGNORECASE)]
#         if matches:
#             name_positions.extend(matches)
#     return name_positions

# Find ages in the clinical letters
def find_ages(clinical_letters):
    age_pattern = r'\b(\d{1,2})(?=\s*(years old|yo|year old|y/o\b))'
    ages = [(m.group(1), (m.start(1), m.end(1))) for m in re.finditer(age_pattern, clinical_letters, re.IGNORECASE)]
    return ages

# Find dates in the clinical letters
def find_dates(clinical_letters):
    date_patterns = [
        r'(?<!\S)\d{4}[/-]\d{2}[/-]\d{2}(?=\s|[.,]|$)',        # Matches complete dates like 2070-12-01 or 2062/04/15
        r'(?<!\S)\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?=\s|[.,]|$)',  # Matches complete dates like 4/66, 2/67, 03/01/64, 4/15/2062
        r'(?<!\S)\w+\s\d{1,2},\s\d{4}(?=\s|[.,]|$)',           # Matches dates like February 2071
        r'(?<!\S)\d{1,2}\s\w+\s\d{4}(?=\s|[.,]|$)',            # Matches dates like 31 December 2070
        r'(?<!\S)\d{1,2}[/-]\d{2}(?=\s|[.,]|$)',               # Specifically for dates like 2/67 or 4/63
        r'(?<!\S)\d{4}(?=\s|[.,]|$)',                          # Matches standalone years like 2065
        r'(?<!\S)(January|February|March|April|May|June|July|August|September|October|November|December)(?=\s|$)'  # Matches standalone month names 
    ]
    dates = []
    for pattern in date_patterns:
        matches = [(m.group(), (m.start(), m.end())) for m in re.finditer(pattern, clinical_letters)]
        dates.extend(matches)
    return dates

# Class for Named Entity Recognition using ClinicalBERT
class NER_ClinicalBERT:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag2idx = {'O': 0, 'ID': 1, 'PHI': 2, 'NAME': 3, 'CONTACT': 4, 'DATE': 5, 'AGE': 6, 'PROFESSION': 7, 'LOCATION': 8, 'PAD': 9}
    tag_values = ["O", "ID", "PHI", "NAME", "CONTACT", "DATE", "AGE", "PROFESSION", "LOCATION", "PAD"]

    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=False)
    model = AutoModelForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=len(tag2idx))

    MAX_LEN = 75
    bs = 4

    def __init__(self):
        if os.path.exists("Models/NER_ClinicalBERT.pt"):
            print("Loading model")
            state_dict = torch.load("Models/NER_ClinicalBERT.pt", map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print("Using pre-trained Clinical BERT model")

    def perform_NER(self, text):
        list_of_sents = sent_tokenize(text)
        list_of_tuples_by_sent = []

        for sent in list_of_sents:
            tokenized_sentence = self.tokenizer.encode(sent, truncation=True)
            input_ids = torch.tensor([tokenized_sentence]).to(self.device)

            with torch.no_grad():
                output = self.model(input_ids)
            label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
            new_tokens, new_labels = [], []
            for token, label_idx in zip(tokens, label_indices[0]):
                if token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(self.tag_values[label_idx])
                    new_tokens.append(token)
            list_of_tuples = [(token, label) for token, label in zip(new_tokens, new_labels)]
            list_of_tuples_by_sent.append(list_of_tuples)

        # Extract start and end positions and combine tokens for the same entity
        clinical_bert_entities = {label: [] for label in self.tag_values if label != "O" and label != "PAD"}
        for sent in list_of_tuples_by_sent:
            current_entity = ""
            current_label = ""
            current_start = -1
            for token, label in sent:
                if label != "O" and label != "PAD":
                    if label == current_label:
                        if current_label == "AGE" and current_entity != "":
                            end = current_start + len(current_entity)
                            clinical_bert_entities[current_label].append((current_entity.strip(), current_start, end))
                            current_entity = token.replace(" ", "")
                            current_start = text.find(token, current_start + 1)
                        else:
                            current_entity += token.replace(" ", "")
                    else:
                        if current_label:
                            end = current_start + len(current_entity)
                            clinical_bert_entities[current_label].append((current_entity.strip(), current_start, end))
                        current_entity = token.replace(" ", "")
                        current_label = label
                        current_start = text.find(token, current_start + 1)
            if current_label:
                end = current_start + len(current_entity)
                clinical_bert_entities[current_label].append((current_entity.strip(), current_start, end))
        return clinical_bert_entities

# Function to load and parse XML annotations
def load_annotations(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    annotations = {'DATE': [], 'NAME': [], 'AGE': [], 'CONTACT': [], 'PROFESSION': [], 'LOCATION': [], 'ID': [], 'PHI': []}
    
    for tag_type in annotations:
        for elem in root.findall(f".//{tag_type}"):
            annotations[tag_type].append({
                'text': elem.attrib['text'],
                'start': int(elem.attrib['start']),
                'end': int(elem.attrib['end'])
            })
    
    text = root.find('.//TEXT').text if root.find('.//TEXT') is not None else ""
    return annotations, text

# Function to compare annotations with detections
def compare_entities(annotations, detections):
    results = {}
    for key in annotations.keys():
        correct_count = 0
        for annotation in annotations[key]:
            for detection in detections[key]:
                if (annotation['text'] == detection[0] and
                    annotation['start'] == detection[1] and
                    annotation['end'] == detection[2]):
                    correct_count += 1
                    break
        results[key] = {'total': len(annotations[key]), 'correctly_detected': correct_count}
    return results

# Merge regex and ClinicalBERT results
def merge_results(clinical_bert_results, regex_results):
    merged_results = {key: [] for key in regex_results.keys()}

    for entity_type in regex_results:
        merged_results[entity_type].extend(regex_results[entity_type])

    for entity_type in clinical_bert_results:
        if entity_type in merged_results:
            merged_results[entity_type].extend(clinical_bert_results[entity_type])

    return merged_results
import pandas as pd

# # Function to extract first names from specified sheets and range in the Excel file
# def extract_first_names_from_excel(file_path, sheet_names, start_row):
#     first_names = []
#     xls = pd.ExcelFile(file_path)
#     for sheet_name in sheet_names:
#         df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
#         # Extract names from the first column starting from the specified row
#         filtered_names = df.iloc[start_row-1:, 0].dropna().astype(str)
#         #filtered_names = filtered_names[filtered_names.str.match(r'^[A-Za-z]+$') & (filtered_names.str.len() < 20)]
#         first_names.extend(filtered_names.tolist())
#     return first_names

# # Function to extract surnames from the second column in the CSV file
# def extract_surnames_from_csv(file_path):
#     df = pd.read_csv(file_path)
#     if 'surname' in df.columns:
#         filtered_surnames = df['surname'].dropna().astype(str)
#         #filtered_surnames = filtered_surnames[filtered_surnames.str.match(r'^[A-Za-z]+$') & (filtered_surnames.str.len() < 20)]
#         return filtered_surnames.tolist()
#     return []

# # Define file paths and parameters
# first_names_file_path = 'dataset/babynames1996to2021 (1).xlsx'
# surnames_file_path = 'dataset/most-common-surnames-multi-year-data.csv'
# sheet_names = ['1', '2']
# start_row = 9

# # Extract first names and surnames
# first_names = extract_first_names_from_excel(first_names_file_path, sheet_names, start_row)
# surnames = extract_surnames_from_csv(surnames_file_path)
# Paths to the files
# first_names_file_path = 'dataset/input/interall.csv'
# surnames_file_path = 'dataset/input/intersurnames.csv'
clinical_letters_file_path = 'dataset/training-PHI-Gold-Set1/220-03.xml'
xml_file_path = 'dataset/training-PHI-Gold-Set1/220-03.xml'

# Load data
# first_names = load_first_column_from_csv(first_names_file_path)
# surnames = load_first_column_from_csv(surnames_file_path)
clinical_letters = load_clinical_letters_from_xml(clinical_letters_file_path)

# # Combine first names and surnames into a single list
# all_names = first_names + surnames

# # Detect potential names using Stanza
# detected_entities_stanza = detect_names_with_stanza(clinical_letters)

# # Verify detected names against the dataset
# verified_names = verify_names(detected_entities_stanza, all_names)

# # Find occurrences and positions of verified names
# name_occurrences = find_name_occurrences(verified_names, clinical_letters)

# Find and display ages
ages = find_ages(clinical_letters)
print("Ages found in clinical letters:\n", ages)

# Find and display dates
dates = find_dates(clinical_letters)
print("Dates found in clinical letters:\n", dates)

# Convert regex results to required format
regex_results = {
    'NAME': [(name, start, end) for name, start, end in name_occurrences],
    'AGE': [(age[0], age[1][0], age[1][1]) for age in ages],
    'DATE': [(date[0], date[1][0], date[1][1]) for date in dates],
    'CONTACT': [],
    'PROFESSION': [],
    'LOCATION': [],
    'ID': [],
    'PHI': []
}

# Perform NER using ClinicalBERT for other entities
ner_clinical_bert = NER_ClinicalBERT()
clinical_bert_results = ner_clinical_bert.perform_NER(clinical_letters)

# Print ClinicalBERT results
print("\nClinicalBERT Results:")
for entity_type, entities in clinical_bert_results.items():
    print(f"{entity_type}:")
    for entity in entities:
        print(f"  {entity}")

# Merge results
merged_results = merge_results(clinical_bert_results, regex_results)

# Load annotations and text from XML
annotations, clinical_text = load_annotations(xml_file_path)

# Compare detected entities with annotations
comparison_results = compare_entities(annotations, merged_results)

# Print regex results
print("\nRegex-based Results:")
for entity_type, entities in regex_results.items():
    print(f"{entity_type}:")
    for entity in entities:
        print(f"  {entity}")

# Print merged results
print("\nMerged Results:")
for entity_type, entities in merged_results.items():
    print(f"{entity_type}:")
    for entity in entities:
        print(f"  {entity}")

# Print the comparison results
print("\nComparison Results:")
for entity_type, result in comparison_results.items():
    print(f"{entity_type}: Total = {result['total']}, Correctly Detected = {result['correctly_detected']}")

# Print loaded annotations for verification
print("\nLoaded Annotations:")
for type_key, items in annotations.items():
    print(type_key, items)
import xml.etree.ElementTree as ET

# Function to load and parse XML annotations
def load_annotations_and_text(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    annotations = {'DATE': [], 'NAME': [], 'AGE': [], 'CONTACT': [], 'PROFESSION': [], 'LOCATION': [], 'ID': [], 'PHI': []}
    
    for tag_type in annotations:
        for elem in root.findall(f".//{tag_type}"):
            annotations[tag_type].append({
                'text': elem.attrib['text'],
                'start': int(elem.attrib['start']),
                'end': int(elem.attrib['end'])
            })
    
    text = root.find('.//TEXT').text if root.find('.//TEXT') is not None else ""
    return annotations, text

# Function to highlight entities in the text
def highlight_entities(text, annotations):
    entities = []
    for entity_list in annotations.values():
        entities.extend(entity_list)
    
    entities = sorted(entities, key=lambda x: x['start'])
    
    highlighted_text = ""
    last_index = 0
    
    for entity in entities:
        start = entity['start']
        end = entity['end']
        highlighted_text += text[last_index:start]
        highlighted_text += f"\033[93m{text[start:end]}\033[0m"
        last_index = end
    
    highlighted_text += text[last_index:]
    
    return highlighted_text

# Function to save the highlighted text to a file
def save_highlighted_text(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
    print(f"Highlighted text saved to {file_path}")

# Define file paths
xml_file_path = 'dataset/training-PHI-Gold-Set1/220-03.xml'
output_file_path = 'highlighted_annotated_text.txt'

# Load annotations and text
annotations, clinical_text = load_annotations_and_text(xml_file_path)

# Highlight entities in the text
highlighted_text = highlight_entities(clinical_text, annotations)

# Save the highlighted text to a file
save_highlighted_text(highlighted_text, output_file_path)
