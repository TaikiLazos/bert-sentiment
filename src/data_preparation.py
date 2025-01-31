import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_json_files(json_dir):
    """Load all JSON files from directory into a list of dictionaries"""
    data = []
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    print(f"Loading {len(json_files)} JSON files...")
    for filename in tqdm(json_files):
        file_path = os.path.join(json_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                # Extract only needed fields
                processed_data = {
                    'topic': json_data.get('topic', ''),
                    'source': json_data.get('source', ''),
                    'bias': json_data.get('bias', 0),
                    'title': json_data.get('title', ''),
                    'content': json_data.get('content', ''),
                    'date': json_data.get('date', ''),
                    'ID': json_data.get('ID', '')
                }
                data.append(processed_data)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return data

def create_dataset_splits(data, output_dir='data/processed'):
    """Create train and test splits and save them"""
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split sizes
    total_samples = len(df)
    train_size = int(0.7 * total_samples)
    test_size = int(0.1 * total_samples)
    
    # Create splits
    train_df = df[:train_size]
    test1_df = df[train_size:train_size + test_size]
    test2_df = df[train_size + test_size:train_size + 2*test_size]
    test3_df = df[train_size + 2*test_size:]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle (more efficient for large datasets)
    print("\nSaving splits...")
    train_df.to_pickle(os.path.join(output_dir, 'train.pkl'))
    test1_df.to_pickle(os.path.join(output_dir, 'test1.pkl'))
    test2_df.to_pickle(os.path.join(output_dir, 'test2.pkl'))
    test3_df.to_pickle(os.path.join(output_dir, 'test3.pkl'))
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test1 samples: {len(test1_df)}")
    print(f"Test2 samples: {len(test2_df)}")
    print(f"Test3 samples: {len(test3_df)}")
    
    # Print bias distribution
    print("\nBias distribution in training set:")
    print(train_df['bias'].value_counts().sort_index())
    
    return train_df, [test1_df, test2_df, test3_df]

def main():
    # Configure paths
    json_dir = 'data/jsons'
    output_dir = 'data/processed'
    
    # Load all JSON files
    data = load_json_files(json_dir)
    
    if not data:
        print("No data loaded. Exiting.")
        return
    
    # Create and save splits
    train_df, test_dfs = create_dataset_splits(data, output_dir)
    
    print("\nData preparation completed!")
    print(f"Files saved in {output_dir}")

if __name__ == "__main__":
    main() 