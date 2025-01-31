# This file is used to transform the json files to train.json and test.json

import json
import os
import random
from typing import Dict, List, Tuple
from tqdm import tqdm

def process_stance(stance_str: str) -> Tuple[int, float]:
    """
    Convert stance string like 'left (51.22%)' to numerical label and confidence
    Returns (label, confidence)
    
    Label mapping:
    2: left with >=80% confidence
    1: left with <80% confidence
    0: center
    -1: right with <80% confidence
    -2: right with >=80% confidence
    """
    if not isinstance(stance_str, str) or stance_str == '?':
        return None, None
    
    try:
        # Split stance and confidence
        stance, conf_str = stance_str.split('(')
        stance = stance.strip().lower()
        confidence = float(conf_str.rstrip('%)'))
        
        if stance == 'center':
            return 0, confidence
        elif stance == 'left':
            return 2 if confidence >= 80 else 1, confidence
        elif stance == 'right':
            return -2 if confidence >= 80 else -1, confidence
        else:
            return None, None
    except:
        return None, None

def transform_json(json_file_path: str, output_dir: str = "data/processed") -> None:
    """Transform the JSON file into train and test sets"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON data
    print(f"Loading data from {json_file_path}...")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Process each entry
    print("Processing entries...")
    processed_data = []
    for item in tqdm(data):
        label, confidence = process_stance(item['stance'])
        if label is not None:  # Skip entries with invalid stance
            processed_item = {
                'text': item['description'],
                'title': item['title'],
                'label': label,
                'confidence': confidence,
                'original_stance': item['stance'],
                'url': item['url'],
                'event_time': item['event_time'],
                'mention_source': item['mention_source']
            }
            processed_data.append(processed_item)
    
    print(f"\nProcessed {len(processed_data)} valid records out of {len(data)} total records")
    
    # Shuffle data
    random.shuffle(processed_data)
    
    # Split data (70% train, 10% each test set)
    n = len(processed_data)
    train_idx = int(n * 0.7)
    test_size = int(n * 0.1)
    
    splits = {
        'train.json': processed_data[:train_idx],
        'test1.json': processed_data[train_idx:train_idx + test_size],
        'test2.json': processed_data[train_idx + test_size:train_idx + 2*test_size],
        'test3.json': processed_data[train_idx + 2*test_size:]
    }
    
    # Save splits and print statistics
    print("\nSaving splits and calculating statistics...")
    label_names = {
        2: "Strong Left (≥80%)",
        1: "Weak Left (<80%)",
        0: "Center",
        -1: "Weak Right (<80%)",
        -2: "Strong Right (≥80%)"
    }
    
    for filename, split_data in splits.items():
        # Save split
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        # Calculate and print statistics
        print(f"\n{filename} statistics:")
        print(f"Total records: {len(split_data)}")
        
        # Label distribution
        label_dist = {}
        for item in split_data:
            label_dist[item['label']] = label_dist.get(item['label'], 0) + 1
        
        print("Label distribution:")
        for label in sorted(label_dist.keys()):
            count = label_dist[label]
            percentage = (count / len(split_data)) * 100
            print(f"{label_names[label]}: {count} ({percentage:.2f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transform GDELT mentions JSON data')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Directory to save processed files')
    
    args = parser.parse_args()
    transform_json(args.input_path, args.output_dir)
