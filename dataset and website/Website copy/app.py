from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from deidentify import calculate_k_anonymity, calculate_unicity, compare_metrics, deidentify_text, deidentify_text1, deidentify_textReplace, extract_entities_with_context, calculate_l_diversity, load_clinical_letters_from_xml
from flask import request, session, jsonify

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key_for_local_dev')  # Set the secret key for session management
UPLOAD_FOLDER = 'uploads'  # Define the folder to save uploaded files

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Set the upload folder in the app configuration

@app.route('/')
def index():
    return render_template('index.html')  # Render the homepage

@app.route('/deidentify', methods=['POST'])
def deidentify():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read the file content
    with open(filepath, 'r') as file:
        original_text = file.read()

    # Retrieve settings from session
    entities_to_deidentify = session.get('entities_to_deidentify', [])
    tools = session.get('tools', {})
    models = session.get('models', {})

    # Debug print statements
    print("Entities to deidentify:", entities_to_deidentify)
    print("Tools:", tools)
    print("Models:", models)

    # Initialize results
    highlighted_texts = []
    deidentified_texts = []
    action = session['action']  # Get the action (redact or replace) from the session
    
    # Process each entity for deidentification
    for entity in entities_to_deidentify:
        tool_name = tools.get(entity, 'default_tool')
        model_name = models.get(entity, 'default_model')

        # Perform deidentification based on the provided action
        highlighted_text, deidentified_text = deidentify_text(
            original_text, action
        )
    
    # Return the deidentified and highlighted text
    return jsonify({
        'originalText': deidentified_text,
        'deidentifiedText': highlighted_text
    })

# Define paths for unique files based on entity types
unique_files = {
    'name': 'unique_names.txt',
    'age': 'unique_ages.txt',
    'date': 'unique_dates.txt',
    'profession': 'unique_professions.txt',
    'location': 'unique_locations.txt'
}

@app.route('/update_and_deidentify', methods=['POST'])
def update_and_deidentify():
    data = request.get_json()
    text = data['text']
    entity_type = data['entity_type']
    sourceText = data['sourceText']
    print(sourceText)
    
    # Append the new text to the appropriate unique file based on entity type
    file_path = unique_files[entity_type]
    with open(file_path, 'a') as file:
        file.write('\n' + text)
    print('done writing')
    
    # Re-run the deidentify function after updating the unique file
    original_text = request.form.get('original_text')
    action = session.get('action')
    highlighted_text, deidentified_text = deidentify_textReplace(sourceText, action)

    # Return the updated deidentified and highlighted text
    return jsonify({
        'originalText': highlighted_text,
        'deidentifiedText': deidentified_text
    })

@app.route('/process_batch', methods=['POST'])
def process_batch():
    files = request.files.getlist('batchInput')  # Get list of files uploaded in the batch
    results = []
    entities_to_deidentify = session.get('entities_to_deidentify', [])
    tools = session.get('tools', {})
    models = session.get('models', {})

    # Debug print statements
    action = session['action']
    all_original_entities = []
    all_replaced_entities = []
    tp, fp, tn, fn = 0, 0, 0, 0  # Initialize true positives, false positives, true negatives, and false negatives

    originalarray = []
    identifiedarray = []

    if action == 'redact':  # If the action is to redact
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print("processing ", filepath)
            with open(filepath, 'r') as file:
                content = file.read()
            clinical_letters = load_clinical_letters_from_xml(content)
            highlighted_text, deidentified_text = deidentify_text(
                content, action)
            results.append({
                'filename': filename,
                'original': highlighted_text,
                'redacted': deidentified_text,
            })
        # Return the results for the redact action
        return jsonify({
            'action': 'redact',
            'results': results,
        })
    else:  # If the action is to replace
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            print("processing ", filepath)
            with open(filepath, 'r') as file:
                content = file.read()
            originalarray.extend(extract_entitiesfromxml(content))
            print(originalarray)
            clinical_letters = load_clinical_letters_from_xml(content)
            highlighted_text, deidentified_text, replaced_text, original_entities, replaced_entities = deidentify_text1(
                content, action
            )
            identifiedarray.extend(original_entities)
            
            # Extract context for original and replaced entities
            original_entities_with_context = extract_entities_with_context(content, original_entities)
            replaced_entities_with_context = extract_entities_with_context(replaced_text, replaced_entities)

            all_original_entities.extend(original_entities_with_context)
            all_replaced_entities.extend(replaced_entities_with_context)
            
            results.append({
                'filename': filename,
                'original': highlighted_text,
                'redacted': deidentified_text,
                'replaced_entities_with_context': replaced_entities_with_context
            })

        # Calculate true positives, false positives, and false negatives
        original_set = set(originalarray)
        identified_set = set(identifiedarray)
        print(len(original_set))
        print("original set-----------------------")
        print(original_set)
        print(len(original_set))
        print("identified_set-----------------------")
        print(identified_set)
        print(len(identified_set))
        tp = len(original_set & identified_set)  # Correctly identified
        fp = len(identified_set - original_set)  # Incorrectly identified
        print(identified_set - original_set)
        fn = len(original_set - identified_set)  # Missed entities
        
        # Calculate TN (true negatives) - generally not used directly in NER evaluation
        tn = 0  # For NER, TN is generally not well-defined

        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Negatives (TN): {tn}")

        # Optionally, calculate and print precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")

        # Aggregate and calculate risk metrics
        original_k_anonymity = calculate_k_anonymity(all_original_entities)
        original_l_diversity = calculate_l_diversity(all_original_entities)
        original_unicity = calculate_unicity(all_original_entities)

        replaced_k_anonymity = calculate_k_anonymity(all_replaced_entities)
        replaced_l_diversity = calculate_l_diversity(all_replaced_entities)
        replaced_unicity = calculate_unicity(all_replaced_entities)

        original_metrics = (original_unicity, original_k_anonymity, original_l_diversity)
        replaced_metrics = (replaced_unicity, replaced_k_anonymity, replaced_l_diversity)
        risk_increased = compare_metrics(original_metrics, replaced_metrics)

        # Determine files with the highest risk based on the replaced entities
        replaced_entities_across_files = []
        for result in results:
            replaced_entities_across_files.extend(result['replaced_entities_with_context'])
        
        # Calculate risk across all files
        file_risks = calculate_risk_across_files(replaced_entities_across_files, results)

        # Sort files by risk level
        sorted_files_by_risk = sorted(file_risks, key=lambda x: x[1], reverse=True)
        print(sorted_files_by_risk)

        # Return results, risk assessment, and files with the highest risk
        return jsonify({
            'action': 'replace',
            'results': results,
            'riskAssessment': {
                'originalMetrics': original_metrics,
                'replacedMetrics': replaced_metrics,
                'riskIncreased': risk_increased
            },
            'filesWithHighRisk': sorted_files_by_risk
        })

# Calculate similarities within replaced entities using BERT embeddings
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def calculate_within_similarities(replaced_entities_with_context):
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    replaced_contexts = [context for _, context, _ in replaced_entities_with_context]

    # Embed the contexts using BERT
    replaced_embeddings = embed_texts(model, tokenizer, replaced_contexts)

    similarities = []
    for i in range(len(replaced_embeddings)):
        for j in range(i + 1, len(replaced_embeddings)):
            similarity = cosine_similarity(replaced_embeddings[i].reshape(1, -1), replaced_embeddings[j].reshape(1, -1))
            similarities.append(similarity[0][0])

    return similarities

import numpy as np

# Calculate true positives, false positives, and false negatives
def calculate_tp_fp_fn(original_entities, replaced_entities):
    tp, fp, fn = 0, 0, 0
    original_set = set(original_entities)
    replaced_set = set(replaced_entities)

    for entity in replaced_set:
        if entity in original_set:
            tp += 1
        else:
            fp += 1

    for entity in original_set:
        if entity not in replaced_set:
            fn += 1

    return tp, fp, fn

# Calculate true negatives (not generally used in NER evaluation)
def calculate_tn(tp, fp, fn, total_entities):
    return total_entities - (tp + fp + fn)

# Print evaluation metrics (precision, recall, F1-score)
def print_evaluation_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

# Calculate risk across files using BERT embeddings and cosine similarity
def calculate_risk_across_files(replaced_entities_with_context, results):
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    replaced_contexts = [context for _, context, _ in replaced_entities_with_context]
    replaced_embeddings = embed_texts(model, tokenizer, replaced_contexts)

    file_risks = []
    for result in results:
        file_contexts = [context for _, context, _ in result['replaced_entities_with_context']]
        file_embeddings = embed_texts(model, tokenizer, file_contexts)

        low_similarity_count = 0
        for file_emb in file_embeddings:
            similarities = cosine_similarity(file_emb.reshape(1, -1), replaced_embeddings)
            # Exclude self-comparison by setting self similarity to 1 (max similarity)
            similarities = np.delete(similarities[0], np.where(similarities[0] == 1.0))
            low_similarity_count += sum(1 for sim in similarities if sim < 0.5)  # Threshold can be adjusted

        file_risks.append((result['filename'], low_similarity_count))
    
    return sorted(file_risks, key=lambda x: x[1], reverse=True)

# Embed texts using BERT model
def embed_texts(model, tokenizer, texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

from collections import Counter

# Extract entities from text (e.g., 'XXX-NAME', 'XXX-DATE', etc.)
def extract_entities(text):
    # Implement logic to extract redacted entities
    # This is a simplified example
    return [word for word in text.split() if word.startswith('XXX-')]

@app.route('/process_deidentification', methods=['POST'])
def process_deidentification():
    # Extract data from form and store in session
    session['entities_to_deidentify'] = request.form.getlist('entities')
    session['tools'] = {}
    session['models'] = {}
    session['files'] = {}
    session['action'] = request.form.get('action')

    # Collect all settings for each entity
    for entity in session['entities_to_deidentify']:
        tool_key = f"{entity.lower()}Tool"
        model_key = f"{entity.lower()}Model"
        file_key = f"{entity.lower()}Dataset"

        session['tools'][entity] = request.form.get(tool_key, 'default_tool')
        session['models'][entity] = request.form.get(model_key, 'default_model')

        # Handle file uploads for datasets
        if file_key in request.files:
            file = request.files[file_key]
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                session['files'][entity] = filepath

    return jsonify({'message': 'Settings saved successfully'})

# Process settings (example function)
def process_settings(settings):
    # Example of processing settings
    for key, value in settings.items():
        if 'dataset' in key and request.files.get(key):
            file = request.files[key]
            file.save(os.path.join('uploads', secure_filename(file.filename)))
        print(f"Setting for {key}: {value}")

    print("Settings processed and saved.")

import xml.etree.ElementTree as ET

# Function to extract entities from XML content
def extract_entitiesfromxml(xml_content):
    # Parse the XML content
    root = ET.fromstring(xml_content)
    
    # Initialize a list to store the extracted entities
    extracted_entities = []

    # Specify the tags of interest
    tags_of_interest = ["NAME", "DATE", "AGE", "LOCATION", "PROFESSION"]

    # Iterate over the tags in the XML
    for tag in root.findall('.//TAGS/*'):
        # Check if the tag itself (not the TYPE attribute) is in the tags of interest
        if tag.tag.upper() in tags_of_interest:
            entity_text = tag.get('text')
            start = int(tag.get('start'))
            end = int(tag.get('end'))
            entity_type = tag.tag.capitalize()  # Use the tag name as the entity type
            extracted_entities.append((entity_text, start, end, entity_type))
    print("getting inside extract")
    print(extracted_entities)
    print("getting out")
    return extracted_entities

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app in debug mode
