import torch
from torch.utils.data import Dataset

class PoliticalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the text
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to the format the model expects
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def get_sample_data():
    """Create sample political texts and labels"""
    sample_texts = [
        "We need complete government control of the economy and wealth redistribution to achieve true equality.",  # far_left
        "Universal healthcare and strong social programs are essential for society's wellbeing.",  # left
        "Both free market principles and social programs have their place in a balanced society.",  # center
        "Lower taxes and reduced government regulation will boost economic growth.",  # right
        "The free market should operate without any government interference whatsoever.",  # far_right
        "Workers must seize the means of production to end capitalist exploitation.",  # far_left
        "Environmental protection and social justice should be our top priorities.",  # left
        "We need practical solutions that balance tradition with progress.",  # center
        "Strong national defense and traditional values make our country great.",  # right
        "Government regulations are destroying our economic freedom and must be eliminated."  # far_right
    ]
    
    # Labels: 0=far_left, 1=left, 2=center, 3=right, 4=far_right
    sample_labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    
    return sample_texts, sample_labels 