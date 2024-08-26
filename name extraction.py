import os
from bs4 import BeautifulSoup

# Define the directory containing the XML files
xml_directory = 'dataset/training-PHI-Gold-Set1'

# Function to extract names from a single XML file using BeautifulSoup
def extract_names_from_xml(xml_file):
    try:
        with open(xml_file, 'r', encoding='utf-8') as file:
            content = file.read()

        soup = BeautifulSoup(content, 'lxml-xml')
        names = set()

        # Debug: Print the entire XML content to ensure it's being read correctly
        print(f"Processing XML content of {xml_file}:")
        
        
        # Extract the text attribute from all NAME tags
        name_tags = soup.find_all('PROFESSION')
        print(f"Found {len(name_tags)} NAME tags in {xml_file}")

        for name_tag in name_tags:
            text = name_tag.get('text')
            print(f"Found NAME tag with text: {text}")
            if text:
                names.add(text)
                if text=="ON":
                    print(xml_file)
                    print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
        return names
    except Exception as e:
        print(f"Error parsing XML file {xml_file}: {e}")
        return set()


# Collect unique names from all XML files
unique_names = set()

# Process each XML file in the directory
for filename in os.listdir(xml_directory):
    if filename.endswith('.xml'):
        file_path = os.path.join(xml_directory, filename)
        print(f"Processing file: {file_path}")
        names = extract_names_from_xml(file_path)
        unique_names.update(names)

# Print unique names
print("Unique names extracted:")
# for name in unique_names:
#     #print(name)

# Optionally, write unique names to a text file
with open('unique_professions.txt', 'w', encoding='utf-8') as f:
    for name in unique_names:
        f.write(f"{name}\n")

print("Unique names have been written to 'unique_names.txt'.")
