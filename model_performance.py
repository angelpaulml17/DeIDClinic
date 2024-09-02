import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
entity_types = ['ID', 'PHI', 'NAME', 'CONTACT', 'DATE', 'AGE', 'PROFESSION', 'LOCATION']
models = ['NER-BERT', 'NER-CRF', 'ClinicalBERT', 'BiLSTM ELMo', 'BioBERT']

# Data
data = {
    'Entity Type': entity_types,
    'NER-BERT': [0.9685, 0.0000, 0.9766, 0.9451, 0.9780, 0.9042, 0.7630, 0.9422],
    'NER-CRF': [0.97, 0.00, 0.96, 0.96, 0.97, 0.94, 0.63, 0.90],
    'ClinicalBERT': [0.9891, 0.0000, 0.97, 0.9536, 0.9708, 0.9512, 0.7297, 0.9206],
    'BiLSTM ELMo': [0.92, 0.00, 0.97, 0.94, 0.96, 0.85, 0.79, 0.91],
    'BioBERT': [0.8974, 0.00, 0.9363, 0.8971, 0.9648, 0.9392, 0.6545, 0.8845]
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(12, 6))
parallel_coordinates(df, 'Entity Type', color=('#556270', '#4ECDC4', '#C7F464', '#FF6B6B', '#C44D58'))

# Add labels and title
plt.title('Parallel Coordinates Plot of F1-Scores Across Entity Types')
plt.xlabel('Entity Type')
plt.ylabel('F1-Score')

# Show plot
plt.show()
