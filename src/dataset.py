import torch
from torch.utils.data import Dataset
import json
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
    def __init__(self, data, tokenizer, max_length=512):
        """
        Initialize dataset with JSON data
        
        Args:
            data: List of dictionaries containing 'text', 'title', 'mention_source', and 'label'
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Label mapping from [-2, -1, 0, 1, 2] to [0, 1, 2, 3, 4]
        self.label_map = {
            -2: 0,  # far_left -> 0
            -1: 1,  # left -> 1
             0: 2,  # center -> 2
             1: 3,  # right -> 3
             2: 4   # far_right -> 4
        }
        
    def preprocess_text(self, text):
        """Preprocess text with lemmatization and stopword removal"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words]
        
        return ' '.join(words)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Preprocess text and title
        processed_text = self.preprocess_text(item['text'])
        processed_title = self.preprocess_text(item['title'])
        
        # Combine features with special tokens
        combined_text = (f"[TEXT] {processed_text} [TITLE] {processed_title} "
                        f"[SOURCE] {item['mention_source']}")
        
        # Tokenize the combined text
        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Map the label to the correct index
        label = self.label_map[item['label']]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dataset(json_path):
    """Load and preprocess the JSON dataset"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} examples")
        return data
    except UnicodeDecodeError as e:
        print("Error: Unable to read the file due to encoding issues.")
        print("Attempting to read with different encoding...")
        try:
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} examples")
            return data
        except Exception as e:
            print(f"Failed to load the file: {e}")
            raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format: {e}")
        raise
    except Exception as e:
        print(f"Error: Unexpected error while loading data: {e}")
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