import seaborn as sns
import matplotlib.pyplot as plt

# Data
entity_types = ['ID', 'PHI', 'NAME', 'CONTACT', 'DATE', 'AGE', 'PROFESSION', 'LOCATION']
models = ['NER-BERT', 'NER-CRF', 'ClinicalBERT', 'BiLSTM ELMo', 'BioBERT']
data = [
    [0.9685, 0.97, 0.9891, 0.92, 0.8974],
    [0.0000, 0.00, 0.0000, 0.00, 0.00],
    [0.9766, 0.96, 0.97, 0.97, 0.9363],
    [0.9451, 0.96, 0.9536, 0.94, 0.8971],
    [0.9780, 0.97, 0.9708, 0.96, 0.9648],
    [0.9042, 0.94, 0.9512, 0.85, 0.9392],
    [0.7630, 0.63, 0.7297, 0.79, 0.6545],
    [0.9422, 0.90, 0.9206, 0.91, 0.8845]
]

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data, annot=True, xticklabels=models, yticklabels=entity_types, cmap='coolwarm', cbar=True)

# Add title
plt.title('Heatmap of F1-Scores for Different Models Across Entity Types')

# Show plot
plt.show()
