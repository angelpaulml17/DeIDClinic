import re
import torch
import numpy as np
from transformers import BertTokenizer, AutoModelForTokenClassification
from nltk.tokenize import sent_tokenize
import stanza
import xml.etree.ElementTree as ET

# Load unique names from a file
def load_unique_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        unique_names = [line.strip() for line in file.readlines()]
    return unique_names

# Find occurrences and positions of unique names
def find_name_occurrences(names, text):
    name_positions = []
    for name in names:
        pattern = rf"\b{name}\b"
        matches = [(name, m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]
        name_positions.extend(matches)
    return name_positions

# Find ages with a more specific context
def find_ages(text):
    age_pattern = r'\b(\d{1,2})\s*(years old|yo|year old|y/o)\b'
    ages = [(m.group(1), (m.start(1), m.end(1))) for m in re.finditer(age_pattern, text, re.IGNORECASE)]
    return ages

# Find dates with improved patterns
def find_dates(text):
    date_patterns = [
        r'\b\d{4}[/-]\d{2}[/-]\d{2}\b',
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},\s\d{4}\b',
        r'\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b'
    ]
    dates = []
    for pattern in date_patterns:
        matches = [(m.group(), (m.start(), m.end())) for m in re.finditer(pattern, text)]
        dates.extend(matches)
    return dates

# Stanza NER for name detection
def detect_names_with_stanza(text):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
    doc = nlp(text)
    detected_entities = [ent.text for ent in doc.ents if ent.type == 'PERSON']
    return detected_entities

# Verify names against a list
def verify_names(detected_entities, names):
    name_parts = set(name.lower() for name in names)
    verified_names = [name for name in detected_entities if name.lower() in name_parts]
    return verified_names

# NER with ClinicalBERT
class NER_ClinicalBERT:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.model = AutoModelForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.model.to(self.device)

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

def load_clinical_letters_from_xml(xml_content):
    # Parse the XML content from a string
    root = ET.fromstring(xml_content)
    clinical_letters = ""
    for text_element in root.findall('.//TEXT'):
        if text_element.text:
            clinical_letters += text_element.text
    return clinical_letters

# Assuming xml_file_path contains the XML content directly
def deidentify_text(xml_content):
    # Load clinical letters from XML content (not from a file path)
    clinical_letters = load_clinical_letters_from_xml(xml_content)

    # Proceed with other operations as before...
    names = load_unique_names("unique_names.txt")
    name_positions = find_name_occurrences(names, clinical_letters)
    ages = find_ages(clinical_letters)
    dates = find_dates(clinical_letters)
    detected_names = detect_names_with_stanza(clinical_letters)
    verified_names = verify_names(detected_names, names)
    ner_clinical_bert = NER_ClinicalBERT()
    bert_entities = ner_clinical_bert.perform_NER(clinical_letters)
    print(bert_entities)
    # Combine and highlight entities
    all_entities = name_positions + ages + dates + verified_names + bert_entities
    highlighted_text = highlight_entities(clinical_letters, all_entities)
    return highlighted_text

def highlight_entities(text, entities):
    for entity in sorted(entities, key=lambda x: x[1], reverse=True):
        start, end = entity[1], entity[2]
        text = text[:start] + "<mark>" + text[start:end] + "</mark>" + text[end:]
    return text
