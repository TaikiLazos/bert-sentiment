import torch
from torch.utils.data import Dataset
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class PoliticalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Initialize dataset with pickle data
        
        Args:
            data_path: Path to pickle file containing DataFrame
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.df = pd.read_pickle(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Combine all fields with special tokens
        combined_text = (f"[TOPIC] {row['topic']} [SOURCE] {row['source']} "
                        f"[TITLE] {row['title']} [CONTENT] {row['content']}")
        
        # Tokenize the combined text
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert bias to label (assuming bias is already in correct format)
        label = row['bias']
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dataset(data_path):
    """Load dataset from pickle file"""
    try:
        df = pd.read_pickle(data_path)
        print(f"Successfully loaded {len(df)} examples from {data_path}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def get_sample_data():
    """Create sample political data in the same format as the JSON file"""
    sample_data = [
        {
            "text": "Government control of economy needed for equality.",
            "title": "Socialist policies for economic justice",
            "label": 2,  # Strong left
            "mention_source": "leftist.news",
            "confidence": 85.5,
            "original_stance": "left (85.5%)"
        },
        {
            "text": "Free market principles drive innovation.",
            "title": "Market freedom essential for growth",
            "label": -2,  # Strong right
            "mention_source": "conservative.com",
            "confidence": 92.3,
            "original_stance": "right (92.3%)"
        },
        {
            "text": "Balanced approach to economic policy needed.",
            "title": "Finding middle ground in policy debate",
            "label": 0,  # Center
            "mention_source": "neutral.org",
            "confidence": 88.7,
            "original_stance": "center (88.7%)"
        }
    ]
    return sample_data

# Update main.py to use the new dataset format:
"""
def train(args):
    # Setup device and model as before...
    
    if args.data_path:
        # Load JSON data
        data = load_dataset(args.data_path)
    else:
        print("Using sample data...")
        data = get_sample_data()
    
    # Split into train and validation
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Create datasets
    train_dataset = PoliticalDataset(train_data, tokenizer)
    val_dataset = PoliticalDataset(val_data, tokenizer)
    
    # Create dataloaders and continue as before...
""" 