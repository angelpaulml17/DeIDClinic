from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizerFast
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_entities_with_context(text, original_entities, window_size=5):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)["offset_mapping"]


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

def calculate_semantic_similarity(entities_with_context):
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    contexts = [context for _, context, _ in entities_with_context]
    embeddings = []

    for context in contexts:
        inputs = tokenizer(context, return_tensors='pt', truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy())

    similarities = cosine_similarity(embeddings)
    return similarities

def calculate_contextual_k_anonymity(entities_with_context, k=2):
    similarities = calculate_semantic_similarity(entities_with_context)

    k_anonymity_count = np.sum(np.sum(similarities > 0.8, axis=1) >= k)
    return k_anonymity_count

# Example usage
text = "Dr. John Smith visited the patient on 2023-05-10 in New York City."
original_entities = [("John Smith", 4, 14, "Name"), ("2023-05-10", 33, 43, "Date"), ("New York City", 47, 60, "Location")]

entities_with_context = extract_entities_with_context(text, original_entities)
k_anonymity_count = calculate_contextual_k_anonymity(entities_with_context, k=2)
print(f"Contextual k-Anonymity count: {k_anonymity_count}")
