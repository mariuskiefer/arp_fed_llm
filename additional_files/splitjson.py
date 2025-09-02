import json
import os
import math

def split_json_file(input_file, output_folder, names):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Calculate the size of each chunk
    total_sentences = len(data)
    chunk_size = math.ceil(total_sentences / len(names))
    
    # Split the data into chunks and save to separate files
    for i, name in enumerate(names):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_sentences)
        chunk = data[start_idx:end_idx]
        
        output_file = os.path.join(output_folder, f"{name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        
        print(f"Created {output_file} with {len(chunk)} sentences")

# Define input and output paths
input_file = "/Users/mariuskiefer/Desktop/arp_fed_llm/fed_sentences_with_entities.json"
output_folder = "/Users/mariuskiefer/Desktop/arp_fed_llm/split_json"
names = ["Adam", "Debi", "Emily", "Ishan", "Marius"]

# Execute the split
split_json_file(input_file, output_folder, names)
