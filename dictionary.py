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
