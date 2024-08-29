import pandas as pd
import re
#import stanza
import xml.etree.ElementTree as ET
from transformers import BertTokenizer, AutoModelForTokenClassification
import torch
from nltk.tokenize import sent_tokenize
import os
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.spec_tokenizers import custom_word_tokenize

from nltk.tokenize.util import align_tokens

# Function to load unique names from a file
def load_unique_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        unique_names = [line.strip() for line in file.readlines()]
    return unique_names

# Function to load unique locations from a file
def load_unique_locations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        unique_locs = [line.strip() for line in file.readlines()]
    return unique_locs

# Function to load unique professions from a file
def load_unique_professions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        unique_prof = [line.strip() for line in file.readlines()]
    return unique_prof

# Function to validate the entities extracted from text
def validate_entities(entities, text):
    validated_entities = []
    for entity in entities:
        if len(entity) == 4:
            word, start, end, entity_type = entity
            validated_entities.append((word, start, end, entity_type))
        else:
            print(f"Unexpected entity format: {entity}")
    return validated_entities

# Function to find occurrences of unique names in clinical letters
def find_name_occurrences(names, clinical_letters):
    name_positions = []
    excluded_words = {"in", "or", "on", "ok", "mi", "p", "do"}  # Words to be excluded

    for name in names:
        if name.lower() in excluded_words:  # Skip excluded words
            continue  # Skip this name
        
        pattern = r'\b{}\b'.format(re.escape(name))  # Create a regex pattern to find the name
        matches = [(name, m.start(), m.end(), 'Name') for m in re.finditer(pattern, clinical_letters, re.IGNORECASE)]
        name_positions.extend(matches)
    return name_positions

# Function to find occurrences of professions in clinical letters
def find_prof_occurrences(names, clinical_letters):
    name_positions = []
    for name in names:
        pattern = r'\b{}\b'.format(re.escape(name))  # Regex pattern to find the profession
        matches = [(name, m.start(), m.end(), 'Profession') for m in re.finditer(pattern, clinical_letters, re.IGNORECASE)]
        name_positions.extend(matches)
    return name_positions

# Function to find occurrences of locations in clinical letters
def find_loc_occurrences(names, clinical_letters):
    name_positions = []
    excluded_words = {"in", "or", "on", "ok", "mi", "p", "do"}  # Words to be excluded

    for name in names:
        if name.lower() in excluded_words:  # Skip excluded words
            continue  # Skip this location
        
        pattern = r'\b{}\b'.format(re.escape(name))  # Regex pattern to find the location
        matches = [(name, m.start(), m.end(), 'Location') for m in re.finditer(pattern, clinical_letters, re.IGNORECASE)]
        name_positions.extend(matches)

    return name_positions

# Function to find ages in clinical letters
def find_ages(clinical_letters):
    age_pattern = r'\b(\d{1,2})(?=\s*(years old|yo|year old|y/o\b))'  # Regex pattern to find ages
    ages = [(m.group(1), m.start(1), m.end(1), 'Age') for m in re.finditer(age_pattern, clinical_letters, re.IGNORECASE)]
    return ages

# Function to find dates in clinical letters
def find_dates(clinical_letters):
    # List of regex patterns to identify different date formats
    date_patterns = [
        r'(?<!\S)\d{4}[/-]\d{2}[/-]\d{2}(?=\s|[.,]|$)', 
        r'(?<!\S)\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?=\s|[.,]|$)',  
        r'(?<!\S)\w+\s\d{1,2},\s\d{4}(?=\s|[.,]|$)', 
        r'(?<!\S)\d{1,2}\s\w+\s\d{4}(?=\s|[.,]|$)', 
        r'(?<!\S)\d{1,2}[/-]\d{2}(?=\s|[.,]|$)', 
        r'(?<!\S)\d{4}(?=\s|[.,]|$)', 
        r'(?<!\S)(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}(?=\s|[.,]|$)'  
    ]
    dates = []
    for pattern in date_patterns:
        matches = [(m.group(), m.start(), m.end(), 'Date') for m in re.finditer(pattern, clinical_letters)]
        dates.extend(matches)
    return dates

# Class for performing Named Entity Recognition (NER) using ClinicalBERT
class NER_ClinicalBERT:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if CUDA is available for GPU usage
    tag2idx = {'O': 0, 'ID': 1, 'PHI': 2, 'NAME': 3, 'CONTACT': 4, 'DATE': 5, 'AGE': 6, 'PROFESSION': 7, 'LOCATION': 8, 'PAD': 9}
    tag_values = ["O", "ID", "PHI", "Name", "CONTACT", "Date", "Age", "Profession", "LOCATION", "PAD"]
    tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', do_lower_case=False)  # Load ClinicalBERT tokenizer
    model = AutoModelForTokenClassification.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels=len(tag2idx))  # Load ClinicalBERT model
    max_length = 512  # Maximum sequence length for BERT

    # Initialize the NER_ClinicalBERT class
    def __init__(self, model_name):
        self.model_name = model_name
        if os.path.exists("../../Models/NER_ClinicalBERT.pt"):
            state_dict = torch.load("../../Models/NER_ClinicalBERT.pt", map_location=self.device)
            self.model.load_state_dict(state_dict)  # Load the model state if available
        else:
            print("Using pre-trained Clinical BERT model")

    # Perform Named Entity Recognition (NER) on the given text
    def perform_NER(self, text, toolname):
        """Implementation of the method that should perform named entity recognition"""
        # Tokenize text into sentences using NLTK
        list_of_sents = sent_tokenize(text)
        list_of_tuples_by_sent = []
        entities_to_deidentify = ['name', 'age', 'location', 'date', 'profession']
        
        # Full text tokenization to handle truncation
        full_text_tokenized = custom_word_tokenize(text, use_bert_tok=True)
        spans = align_tokens(full_text_tokenized, text)
        n = 0
        point = 0 
        all_entities=[]

        # Iterate over each sentence
        for sent in list_of_sents:
            tokenized_sentence = self.tokenizer.encode(sent, truncation=True)  # Tokenize the sentence using BERT tokenizer
            input_ids = torch.tensor([tokenized_sentence])

            # Run inference/classification
            with torch.no_grad():
                output = self.model(input_ids)
            label_indices = np.argmax(output[0].to("cpu").numpy(), axis=2)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
            new_tokens, new_labels, new_start, new_end = [], [], [], []

            # Iterate over tokens and labels
            for token, label_idx in zip(tokens, label_indices[0]):
                # Remove special tokens
                if (token == '[CLS]') or (token == '[SEP]') or (token == "'") or (token == "\"") or (token == "`"):
                    continue
                elif token.startswith("##"):  # Handle subword tokens
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(self.tag_values[label_idx])
                    new_tokens.append(token)
            
            new_start, new_end = zip(*align_tokens(new_tokens, text[point:]))
            new_start, new_end = [point + st for st in new_start], [point + en for en in new_end]

            len_tok_sent = len(custom_word_tokenize(sent, use_bert_tok=True))
            seq_toks = full_text_tokenized[n: n + len_tok_sent]
            seq_start, seq_end = zip(*spans[n: n + len_tok_sent])
            seq_labels = ['O'] * len_tok_sent
            ori_seq = list(zip(seq_toks, seq_start, seq_end))
           
            list_of_tuples = [(token, st, en, 'O') for token, st, en in zip(seq_toks, seq_start, seq_end)]

            # Align tokens with labels
            for token, label, st, en in zip(new_tokens, new_labels, new_start, new_end):
                if tuple([token, st, en]) in ori_seq and label.lower() in entities_to_deidentify:
                    idx = ori_seq.index(tuple([token, st, en]))
                    list_of_tuples[idx] = (token, st, en, label)
            n = n + len_tok_sent

            # Combine consecutive tokens belonging to the same entity type
            combined_entities = []
            current_entity = None
            for token, st, en, label in list_of_tuples:
                if label != 'O' and (current_entity is None or current_entity[3] != label or current_entity[2] < st):
                    if current_entity is not None:
                        combined_entities.append(tuple(current_entity))
                    current_entity = [token, st, en, label]
                elif current_entity is not None and current_entity[3] == label:
                    if current_entity[2] == st:
                        current_entity[0] += token
                    elif current_entity[2] == st - 1:
                        current_entity[0] += " " + token
                    current_entity[2] = en
                else:
                    combined_entities.append((token, st, en, 'O'))

            if current_entity is not None:
                combined_entities.append(tuple(current_entity))

            combined_entities = [entity for entity in combined_entities if entity[3] != 'O']
            all_entities.extend(combined_entities)

            point = point + len(sent)
        
        # Filter and merge final entities
        final_entities = [(token, st, en, label) for token, st, en, label in all_entities if label.lower() in entities_to_deidentify]

        # Merge entities with overlapping start and end positions
        merged_entities = []
        i = 0
        while i < len(final_entities):
            entity = final_entities[i]
            j = i + 1
            while j < len(final_entities) and final_entities[j][3] == entity[3]:
                if final_entities[j][1] == entity[2]:
                    entity = (entity[0] + final_entities[j][0], entity[1], final_entities[j][2], entity[3])
                elif final_entities[j][1] == entity[2] + 1:
                    entity = (entity[0] + " " + final_entities[j][0], entity[1], final_entities[j][2], entity[3])
                else:
                    break
                j += 1
            merged_entities.append(entity)
            i = j

        return merged_entities

# Color mapping for different entity types
color_map = {
    'Name': '#ffc5d9',
    'Age': '#c2f2d0',
    'Date': '#ffcb85',
    'Profession': 'brown',
    'Location': 'green'
}

# Function to highlight replaced entities in the text with colors
def highlight_replaced_entities(text, replaced_entities):
    result = text
    # Sort entities in reverse order to avoid indexing issues when replacing
    sorted_replaced_entities = sorted(replaced_entities, key=lambda x: x[1], reverse=True)
    for entity, start, end, entity_type in sorted_replaced_entities:
        color = color_map.get(entity_type, 'grey')  # Default to grey if no specific color
        result = result[:start] + f"<mark style='background-color: {color}'>{entity}</mark>" + result[end:]

    return result

# Function to highlight original entities in the text with colors
def highlight_entities(text, entities):
    # Initialize character list for highlighting
    chars = [{'char': c, 'highlight': None} for c in text]

    # Mark characters to be highlighted with the specific color
    for entity, start, end, entity_type in entities:
        color = color_map.get(entity_type, 'grey')  # Default to grey if no specific color
        for i in range(start, end):
            chars[i]['highlight'] = color

    # Build the resulting string with colors
    result = ""
    current_color = None
    for char in chars:
        if char['highlight'] and char['highlight'] != current_color:
            if current_color is not None:
                result += "</mark>"
            result += f"<mark style='background-color: {char['highlight']}'>"
            current_color = char['highlight']
        elif not char['highlight'] and current_color is not None:
            result += "</mark>"
            current_color = None
        result += char['char']

    if current_color:
        result += "</mark>"

    return result

import logging

# Function to redact entities from the text
def redact_entities(text, entities):
    logging.basicConfig(level=logging.DEBUG)
    sorted_entities = sorted(entities, key=lambda x: x[1], reverse=True)
    last_end = len(text)

    for entity, start, end, entity_type in sorted_entities:
        logging.debug(f"Processing {entity} from {start} to {end} as {entity_type}")
        if start < last_end:  # Ensure no overlap with previously redacted sections
            color = color_map.get(entity_type, 'grey')  # Use the same color mapping
            redaction = f"<mark style='background-color: {color}'>XXX-{entity_type}</mark>"
            text = text[:start] + redaction + text[end:]
            last_end = start
            logging.debug(f"Redacted {entity_type} at {start}-{end} with {color}")

    return text

# Function to load clinical letters from XML content
def load_clinical_letters_from_xml(xml_content):
    root = ET.fromstring(xml_content)
    clinical_letters = ""
    for text_element in root.findall('.//TEXT'):
        clinical_letters += text_element.text
    return clinical_letters

import random
from flask import session

# Function to replace entities in the text with mapped replacements
def replace_entities(text, entities, replacement_dict):
    new_entities = []
    offset = 0
    for entity, start, end, entity_type in sorted(entities, key=lambda x: x[1]):
        replacement = replacement_dict.get((entity, entity_type))
        if replacement:
            text = text[:start + offset] + replacement + text[end + offset:]
            new_end = start + offset + len(replacement)
            new_entities.append((replacement, start + offset, new_end, entity_type))
            offset += len(replacement) - (end - start)
    return text, new_entities

# Function to create a replacement mapping for entities and save it in a JSON file
def create_replacement_mapping(entities, entity_type, replacement_list, replacement_file='replacements.json'):
    """Create replacement mapping for entities and save them in a JSON file."""
    replacement_dict = load_replacements(replacement_file)
    used_replacements = set(replacement_dict.values())

    for entity, start, end, entity_type in entities:
        key = (entity, entity_type)
        if key not in replacement_dict:
            replacement = random.choice(replacement_list)
            while replacement in used_replacements:
                replacement = random.choice(replacement_list)
            replacement_dict[key] = replacement
            used_replacements.add(replacement)

    save_replacements(replacement_file, replacement_dict)
    return replacement_dict

from datetime import datetime, timedelta

# Function to generate a random date within a range based on the original date
def generate_random_date_within_range(original_date):
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%m/%d/%y',
        '%B %d, %Y',
        '%d %B %Y',
        '%m/%d',
        '%m/%y',
        '%Y',
        '%B'
    ]
    for fmt in formats:
        try:
            parsed_date = datetime.strptime(original_date, fmt)
            year_change = random.randint(-2, 2)
            month_change = random.randint(-3, 3)
            day_change = random.randint(-5, 5)
            new_date = parsed_date + timedelta(days=day_change)
            new_date = new_date.replace(
                year=parsed_date.year + year_change,
                month=max(1, min(12, parsed_date.month + month_change))
            )
            return new_date.strftime(fmt)
        except ValueError:
            continue
    return original_date

# Function to generate a random age within a range based on the original age
def generate_random_age_within_range(original_age):
    try:
        age = int(original_age)
        new_age = age + random.randint(-5, 5)
        return str(max(0, new_age))  # Ensure age is not negative
    except Exception as e:
        print(f"Error generating random age: {e}")
    return original_age

# Function to create a replacement mapping for dates and ages, saving it in a JSON file
def create_replacement_mapping1(entities, replacement_file='replacements.json'):
    """Create replacement mapping for entities and save them in a JSON file."""
    replacement_dict = load_replacements(replacement_file)
    used_replacements = set(replacement_dict.values())

    for entity, start, end, entity_type in entities:
        key = (entity, entity_type)
        if key not in replacement_dict:
            if entity_type == 'Date':
                replacement = generate_random_date_within_range(entity)
            elif entity_type == 'Age':
                replacement = generate_random_age_within_range(entity)
            else:
                continue  # Skip any entity types not handled
                
            replacement_dict[key] = replacement
            used_replacements.add(replacement)

    save_replacements(replacement_file, replacement_dict)
    return replacement_dict

import json
import os

# Function to load names from a CSV file
def load_names_from_csv(file_path):
    df = pd.read_csv(file_path)
    surnames_list = df['Surname'].dropna().unique().tolist()
    full_names_list = df[['FirstName', 'Surname']].dropna().apply(lambda row: f"{row['FirstName']} {row['Surname']}", axis=1).tolist()
    return surnames_list, full_names_list

# Function to load existing replacements from a JSON file
def load_replacements(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                replacements = json.load(file)
                return {tuple(k.split('_')): v for k, v in replacements.items()}
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from {file_path}: {e}")
            return {}
    return {}

# Function to save replacements to a JSON file
def save_replacements(file_path, replacements):
    with open(file_path, 'w', encoding='utf-8') as file:
        replacements_str_keys = {'_'.join(k): v for k, v in replacements.items()}
        json.dump(replacements_str_keys, file, ensure_ascii=False, indent=4)

# Function to create a replacement mapping for names, saving it in a JSON file
def create_replacement_mappingName(entities, entity_type, surnames_list, full_names_list, replacement_file):
    # Load existing replacements
    print('loadreplce')
    replacement_dict = load_replacements(replacement_file)
    used_replacements = set(replacement_dict.values())
    print('loadedreplace')
    for entity, start, end, entity_type in entities:
        if (entity, entity_type) not in replacement_dict:
            entity_words = entity.split()
            if len(entity_words) == 1:
                replacement = random.choice(surnames_list)
                while replacement in used_replacements:
                    replacement = random.choice(surnames_list)
            else:
                full_name = random.choice(full_names_list).split()
                while tuple(full_name) in used_replacements:
                    full_name = random.choice(full_names_list).split()
                replacement = ' '.join(full_name)
            
            replacement_dict[(entity, entity_type)] = replacement
            used_replacements.add(replacement if len(entity_words) == 1 else tuple(full_name))
    print('done check')
    save_replacements(replacement_file, replacement_dict)
    print('dome save')
    return replacement_dict

# Function to convert a string to camelCase
def to_camel_case(s):
    components = s.split('_')
    return components[0].lower() + ''.join(x.title() for x in components[1:])

# Function to combine entities from multiple sources and remove duplicates
def combine_entities(all_entities, filtered_entities):
    # Combine both entity lists
    combined_entities = all_entities + filtered_entities
    
    # Remove duplicates by converting to a set of tuples, which inherently removes duplicates, then back to a list
    unique_entities = list({(ent, start, end, (ent_type.lower()).capitalize()) for (ent, start, end, ent_type) in combined_entities})
    
    # Sort entities based on their start position
    unique_entities.sort(key=lambda x: x[1])
    
    return unique_entities

# Function to de-identify text using various models and tools
def deidentify_text(xml_file_path, action):
    # Load clinical letters from XML file
    clinical_letters = load_clinical_letters_from_xml(xml_file_path)
    all_entities = []
    # Retrieve settings from session
    entities_to_deidentify = session.get('entities_to_deidentify', [])
    tools = session.get('tools', {})
    models = session.get('models', {})
    files = session.get('files', {})
    # print(entities_to_deidentify)
    # print(tools)
    # print(models)
    # print(files)
    replacement_mapping = {}
    replacement_file = 'replacements.json'

    # Detect entities based on user preferences
    for entity_type in entities_to_deidentify:
        tool_name = tools.get(entity_type, 'default_tool')
        model_name = models.get(entity_type, 'default_model')
        dataset_file = files.get(entity_type)
        print('--------------------------')
        print(model_name)
        surnames_list, full_names_list = load_names_from_csv('adjusted-name-combinations-list.csv')
        
        if entity_type == 'Name':
            # Load names from provided file or use default
            names = load_unique_names(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_names.txt")
            name_positions = find_name_occurrences(names, clinical_letters)
            all_entities.extend(name_positions)
            replacement_mapping.update(create_replacement_mappingName(name_positions, "Name", surnames_list, full_names_list, replacement_file))
            print(replacement_mapping)
        
        elif entity_type == 'Age':
            ages = find_ages(clinical_letters)
            all_entities.extend(ages)
            replacement_mapping.update(create_replacement_mapping1(ages))
        
        elif entity_type == 'Date':
            dates = find_dates(clinical_letters)
            all_entities.extend(dates)
            replacement_mapping.update(create_replacement_mapping1(dates))
        
        elif entity_type == 'Profession':
            # Load names from provided file or use default
            names = load_unique_professions(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_professions.txt")
            name_positions = find_prof_occurrences(names, clinical_letters)
            all_entities.extend(name_positions)
            replacement_mapping.update(create_replacement_mapping(name_positions, 'Profession', names))
        
        elif entity_type == 'Location':
            # Load names from provided file or use default
            names = load_unique_locations(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_locations.txt")
            name_positions = find_loc_occurrences(names, clinical_letters)
            all_entities.extend(name_positions)
            replacement_mapping.update(create_replacement_mapping(name_positions, 'Location', names))
        
        # Add logic for other entities as necessary

        # Using NER model specific to the tool selected for each entity
    if 'clinicalbert' in model_name:
            # Assuming an NER function tailored for different models
        ner_model = NER_ClinicalBERT(model_name)
        ner_model_entities = ner_model.perform_NER(clinical_letters, tool_name)
        
        # Filter entities based on type
        filtered_entities = [(ent, start, end, ent_type) for (ent, start, end, ent_type) in ner_model_entities if ent_type.lower() in [etype.lower() for etype in entities_to_deidentify]]
        for ent, start, end, ent_type in filtered_entities:
            if ent_type.lower() == 'name':
                replacement_mapping.update(create_replacement_mappingName([(ent, start, end, ent_type)], "Name", surnames_list, full_names_list, replacement_file))
            elif ent_type.lower() == 'age':
                replacement_mapping.update(create_replacement_mapping1([(ent, start, end, ent_type)]))
            elif ent_type.lower() == 'date':
                replacement_mapping.update(create_replacement_mapping1([(ent, start, end, ent_type)]))
            elif ent_type.lower() == 'profession':
                profession_names = load_unique_professions(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_professions.txt")
                replacement_mapping.update(create_replacement_mapping([(ent, start, end, ent_type)], 'Profession', profession_names, replacement_file))
            elif ent_type.lower() == 'location':
                location_names = load_unique_locations(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_locations.txt")
                replacement_mapping.update(create_replacement_mapping([(ent, start, end, ent_type)], 'Location', location_names, replacement_file))

        print("++++++++++++++++++++++++++++++++by bert++++++++++++++++++++++++++++++++++")
        print(filtered_entities)
        print("++++++++++++++++++++++++++++++++replacements++++++++++++++++++++++++++++++++++")
        print(replacement_mapping)
            #all_entities.extend(filtered_entities)
    #print(all_entities)
    
    # Validate and classify entities
    all_entities1 = combine_entities(all_entities, filtered_entities)
    validated_entities = validate_entities(all_entities1, clinical_letters)

    # Generate highlighted and redacted texts
    highlighted_text = highlight_entities(clinical_letters, validated_entities)
    redacted_text = redact_entities(clinical_letters, validated_entities)
    # print(highlighted_text)
    # print(replacement_mapping)
    
    # Based on the action, either redact or replace the entities
    if action == 'redact':
        highlighted_text = highlight_entities(clinical_letters, validated_entities)
        redacted_text = redact_entities(clinical_letters, validated_entities)
        return highlighted_text, redacted_text

    elif action == 'replace':
        replaced_text, new_entities = replace_entities(clinical_letters, validated_entities, replacement_mapping)
        original_entities = [(ent, start, end, ent_type) for ent, start, end, ent_type in validated_entities]
        replaced_entities = [(replacement_mapping.get((ent, ent_type), ent), start, end, ent_type) for ent, start, end, ent_type in validated_entities]
        
        highlighted_replaced_text = highlight_replaced_entities(replaced_text, new_entities)
       # risk_assessment_results = perform_risk_assessment(highlighted_replaced_text, original_entities, replaced_entities)
       # print(risk_assessment_results)
        return highlighted_text, highlighted_replaced_text

# Function to de-identify text using different methods (redact or replace)
def deidentify_textReplace(clinical_letters, action):
    # Load clinical letters from XML file
    print(clinical_letters)
    all_entities = []

    # Retrieve settings from session
    entities_to_deidentify = session.get('entities_to_deidentify', [])
    tools = session.get('tools', {})
    models = session.get('models', {})
    files = session.get('files', {})
    # print(entities_to_deidentify)
    # print(tools)
    # print(models)
    # print(files)
    replacement_mapping = {}
    replacement_file = 'replacements.json'

    # Detect entities based on user preferences
    for entity_type in entities_to_deidentify:
        tool_name = tools.get(entity_type, 'default_tool')
        model_name = models.get(entity_type, 'default_model')
        dataset_file = files.get(entity_type)
        print('--------------------------')
        print(model_name)
        surnames_list, full_names_list = load_names_from_csv('adjusted-name-combinations-list.csv')

        if entity_type == 'Name':
            # Load names from provided file or use default
            print('loading')
            names = load_unique_names(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_names.txt")
            print('loaded')
            name_positions = find_name_occurrences(names, clinical_letters)
            print('nameposition')
            all_entities.extend(name_positions)
            print('started replacement')
            replacement_mapping.update(create_replacement_mappingName(name_positions, "Name", surnames_list, full_names_list, replacement_file))
            print(replacement_mapping)
        
        elif entity_type == 'Age':
            ages = find_ages(clinical_letters)
            all_entities.extend(ages)
            replacement_mapping.update(create_replacement_mapping1(ages))
        
        elif entity_type == 'Date':
            dates = find_dates(clinical_letters)
            all_entities.extend(dates)
            replacement_mapping.update(create_replacement_mapping1(dates))
        
        elif entity_type == 'Profession':
            # Load names from provided file or use default
            names = load_unique_professions(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_professions.txt")
            name_positions = find_prof_occurrences(names, clinical_letters)
            all_entities.extend(name_positions)
            replacement_mapping.update(create_replacement_mapping(name_positions, 'Profession', names))
        
        elif entity_type == 'Location':
            # Load names from provided file or use default
            names = load_unique_locations(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_locations.txt")
            name_positions = find_loc_occurrences(names, clinical_letters)
            all_entities.extend(name_positions)
            replacement_mapping.update(create_replacement_mapping(name_positions, 'Location', names))
        
        # Add logic for other entities as necessary

        # Using NER model specific to the tool selected for each entity
    if 'clinicalbert' in model_name:
            # Assuming an NER function tailored for different models
        ner_model = NER_ClinicalBERT(model_name)
        ner_model_entities = ner_model.perform_NER(clinical_letters, tool_name)
        
        # Filter entities based on type
        filtered_entities = [(ent, start, end, ent_type) for (ent, start, end, ent_type) in ner_model_entities if ent_type.lower() in [etype.lower() for etype in entities_to_deidentify]]
        
        for ent, start, end, ent_type in filtered_entities:
            if ent_type.lower() == 'name':
                replacement_mapping.update(create_replacement_mappingName([(ent, start, end, ent_type)], "Name", surnames_list, full_names_list, replacement_file))
            elif ent_type.lower() == 'age':
                replacement_mapping.update(create_replacement_mapping1([(ent, start, end, ent_type)]))
            elif ent_type.lower() == 'date':
                replacement_mapping.update(create_replacement_mapping1([(ent, start, end, ent_type)]))
            elif ent_type.lower() == 'profession':
                profession_names = load_unique_professions(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_professions.txt")
                replacement_mapping.update(create_replacement_mapping([(ent, start, end, ent_type)], 'Profession', profession_names, replacement_file))
            elif ent_type.lower() == 'location':
                location_names = load_unique_locations(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_locations.txt")
                replacement_mapping.update(create_replacement_mapping([(ent, start, end, ent_type)], 'Location', location_names, replacement_file))

        print("++++++++++++++++++++++++++++++++by bert++++++++++++++++++++++++++++++++++")
        print(filtered_entities)
        print("++++++++++++++++++++++++++++++++replacements++++++++++++++++++++++++++++++++++")
        print(replacement_mapping)
        #print(filtered_entities)
            #all_entities.extend(filtered_entities)
    #print(all_entities)
    
    # Validate and classify entities
    all_entities1 = combine_entities(all_entities, filtered_entities)
    validated_entities = validate_entities(all_entities1, clinical_letters)

    # Generate highlighted and redacted texts
    highlighted_text = highlight_entities(clinical_letters, validated_entities)
    redacted_text = redact_entities(clinical_letters, validated_entities)
    # print(highlighted_text)
    # print(replacement_mapping)
    
    # Based on the action, either redact or replace the entities
    if action == 'redact':
        highlighted_text = highlight_entities(clinical_letters, validated_entities)
        redacted_text = redact_entities(clinical_letters, validated_entities)
        return highlighted_text, redacted_text

    elif action == 'replace':
        replaced_text, new_entities = replace_entities(clinical_letters, validated_entities, replacement_mapping)
        original_entities = [(ent, start, end, ent_type) for ent, start, end, ent_type in validated_entities]
        replaced_entities = [(replacement_mapping.get((ent, ent_type), ent), start, end, ent_type) for ent, start, end, ent_type in validated_entities]
        
        highlighted_replaced_text = highlight_replaced_entities(replaced_text, new_entities)
       # risk_assessment_results = perform_risk_assessment(highlighted_replaced_text, original_entities, replaced_entities)
       # print(risk_assessment_results)
        return highlighted_text, highlighted_replaced_text

# Function to de-identify text using different methods (redact or replace)- for batch processing on replace
def deidentify_text1(xml_file_path, action):
    # Load clinical letters from XML file
    clinical_letters = load_clinical_letters_from_xml(xml_file_path)
    all_entities = []

    # Retrieve settings from session
    entities_to_deidentify = session.get('entities_to_deidentify', [])
    tools = session.get('tools', {})
    models = session.get('models', {})
    files = session.get('files', {})
    # print(entities_to_deidentify)
    # print(tools)
    # print(models)
    # print(files)
    replacement_mapping = {}
    surnames_list, full_names_list = load_names_from_csv('adjusted-name-combinations-list.csv')
    replacement_file = 'replacements.json'

    # Detect entities based on user preferences
    for entity_type in entities_to_deidentify:
        tool_name = tools.get(entity_type, 'default_tool')
        model_name = models.get(entity_type, 'default_model')
        dataset_file = files.get(entity_type)
        print('--------------------------')
        print(model_name)
        
        if entity_type == 'Name':
            # Load names from provided file or use default
            names = load_unique_names(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_names.txt")
            name_positions = find_name_occurrences(names, clinical_letters)
            all_entities.extend(name_positions)
            replacement_mapping.update(create_replacement_mappingName(name_positions, "Name", surnames_list, full_names_list, replacement_file))
            print(replacement_mapping)
        
        elif entity_type == 'Age':
            ages = find_ages(clinical_letters)
            all_entities.extend(ages)
            replacement_mapping.update(create_replacement_mapping1(ages))
        
        elif entity_type == 'Date':
            dates = find_dates(clinical_letters)
            all_entities.extend(dates)
            replacement_mapping.update(create_replacement_mapping1(dates))
        
        elif entity_type == 'Profession':
            # Load names from provided file or use default
            names = load_unique_professions(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_professions.txt")
            name_positions = find_prof_occurrences(names, clinical_letters)
            all_entities.extend(name_positions)
            replacement_mapping.update(create_replacement_mapping(name_positions, 'Profession', names))
        
        elif entity_type == 'Location':
            # Load names from provided file or use default
            names = load_unique_locations(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_locations.txt")
            name_positions = find_loc_occurrences(names, clinical_letters)
            all_entities.extend(name_positions)
            replacement_mapping.update(create_replacement_mapping(name_positions, 'Location', names))
        
        # Add logic for other entities as necessary

        # Using NER model specific to the tool selected for each entity
    if 'clinicalbert' in model_name:
            # Assuming an NER function tailored for different models
        ner_model = NER_ClinicalBERT(model_name)
        ner_model_entities = ner_model.perform_NER(clinical_letters, tool_name)
        
        # Filter entities based on type
        filtered_entities = [(ent, start, end, ent_type) for (ent, start, end, ent_type) in ner_model_entities if ent_type.lower() in [etype.lower() for etype in entities_to_deidentify]]
        for ent, start, end, ent_type in filtered_entities:
            if ent_type.lower() == 'name':
                replacement_mapping.update(create_replacement_mappingName([(ent, start, end, ent_type)], "Name", surnames_list, full_names_list, replacement_file))
            elif ent_type.lower() == 'age':
                replacement_mapping.update(create_replacement_mapping1([(ent, start, end, ent_type)]))
            elif ent_type.lower() == 'date':
                replacement_mapping.update(create_replacement_mapping1([(ent, start, end, ent_type)]))
            elif ent_type.lower() == 'profession':
                profession_names = load_unique_professions(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_professions.txt")
                replacement_mapping.update(create_replacement_mapping([(ent, start, end, ent_type)], 'Profession', profession_names, replacement_file))
            elif ent_type.lower() == 'location':
                location_names = load_unique_locations(dataset_file if dataset_file and os.path.exists(dataset_file) else "unique_locations.txt")
                replacement_mapping.update(create_replacement_mapping([(ent, start, end, ent_type)], 'Location', location_names, replacement_file))

        print("++++++++++++++++++++++++++++++++by bert++++++++++++++++++++++++++++++++++")
        print(filtered_entities)
        print("++++++++++++++++++++++++++++++++replacements++++++++++++++++++++++++++++++++++")
        print(replacement_mapping)
        #print(filtered_entities)
            #all_entities.extend(filtered_entities)
    #print(all_entities)
    
    # Validate and classify entities
    all_entities1 = combine_entities(all_entities, filtered_entities)
    validated_entities = validate_entities(all_entities1, clinical_letters)

    # Generate highlighted and redacted texts
    highlighted_text = highlight_entities(clinical_letters, validated_entities)
    redacted_text = redact_entities(clinical_letters, validated_entities)
    # print(highlighted_text)
    # print(replacement_mapping)
    
    # Based on the action, either redact or replace the entities
    if action == 'redact':
        highlighted_text = highlight_entities(clinical_letters, validated_entities)
        redacted_text = redact_entities(clinical_letters, validated_entities)
        return highlighted_text, redacted_text

    elif action == 'replace':
        replaced_text, new_entities = replace_entities(clinical_letters, validated_entities, replacement_mapping)
        original_entities = [(ent, start, end, ent_type) for ent, start, end, ent_type in validated_entities]
        replaced_entities = [(replacement_mapping.get((ent, ent_type), ent), start, end, ent_type) for ent, start, end, ent_type in validated_entities]
        
        highlighted_replaced_text = highlight_replaced_entities(replaced_text, new_entities)
       # risk_assessment_results = perform_risk_assessment(highlighted_replaced_text, original_entities, replaced_entities)
       # print(risk_assessment_results)
        return highlighted_text, highlighted_replaced_text, replaced_text, original_entities, replaced_entities
    
import pandas as pd
from collections import Counter

from transformers import BertTokenizerFast

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Function to extract entities with surrounding context from text
def extract_entities_with_context(text, original_entities, window_size=5):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)["offset_mapping"]

    # Convert character index to token index
    def char_to_token_index(char_index):
        for token_index, (start, end) in enumerate(token_offsets):
            if start <= char_index < end:
                return token_index
        return -1

    entities_with_context = []
    for entity, start, end, entity_type in original_entities:
        start_token_index = char_to_token_index(start)
        end_token_index = char_to_token_index(end)

        if start_token_index == -1 or end_token_index == -1:
            continue

        context_start = max(0, start_token_index - window_size)
        context_end = min(len(tokens), end_token_index + window_size)

        context = ' '.join(tokens[context_start:context_end])
        entities_with_context.append((entity, context, entity_type))

    return entities_with_context

# Function to calculate k-anonymity of the entities
def calculate_k_anonymity(entities):
    counts = Counter([context for entity, context, entity_type in entities])
    #print(counts.values())
    k_anonymity = min(counts.values())
    return k_anonymity

# Function to calculate l-diversity of the entities
def calculate_l_diversity(entities):
    counts = Counter([context for entity, context, entity_type in entities])
    l_diversity = len(counts) / len(entities) if entities else 0
    return l_diversity

# Function to calculate unicity of the entities
def calculate_unicity(entities):
    counts = Counter([context for entity, context, entity_type in entities])
    unicity = sum(1 for count in counts.values() if count == 1) / len(entities) if entities else 0
    return unicity

# Function to compare risk metrics between original and replaced entities
def compare_metrics(original_metrics, replaced_metrics):
    risk_increased = replaced_metrics[0] < original_metrics[0] or \
                     replaced_metrics[1] > original_metrics[1]
    return risk_increased
