import argparse
import warnings
# Ignore specific warnings
warnings.filterwarnings('ignore', category=UserWarning)  # Ignore UserWarnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Ignore FutureWarnings
warnings.filterwarnings('ignore', message='.*numpy.*')  # Ignore numpy-related warnings

from train import setup_model, train_model, setup_device
from dataset import PoliticalDataset, get_sample_data
from torch.utils.data import DataLoader
from evaluate import load_model, predict_stance
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Political Stance Classification')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True,
                       help='Whether to train or evaluate the model')
    parser.add_argument('--model_path', type=str, default='best_model.pt',
                       help='Path to save/load model')
    parser.add_argument('--data_path', type=str, 
                       help='Path to dataset (optional, uses sample data if not provided)')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                       help='Name of the pretrained model to use')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    return parser.parse_args()

def train(args):
    # Setup device
    device = setup_device()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model(model_name=args.model_name)
    model = model.to(device)
    
    # Get data (either from file or sample data)
    if args.data_path:
        df = pd.read_csv(args.data_path)
        train_texts = df['content'].tolist()
        train_labels = df['bias'].tolist()
    else:
        print("Using sample data...")
        train_texts, train_labels = get_sample_data()
    
    # Split into train and validation
    split_idx = int(len(train_texts) * 0.8)
    
    # Create datasets
    train_dataset = PoliticalDataset(
        train_texts[:split_idx],
        train_labels[:split_idx],
        tokenizer
    )
    val_dataset = PoliticalDataset(
        train_texts[split_idx:],
        train_labels[split_idx:],
        tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Train the model
    train_model(
        model, 
        train_loader, 
        val_loader, 
        device,
        model_path=args.model_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    print(f"Model saved to {args.model_path}")

def evaluate(args):
    if not args.model_path:
        raise ValueError("Model path must be provided for evaluation")
    
    # Setup device
    device = setup_device()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(
        args.model_path,
        model_name=args.model_name,
        device=device
    )
    
    # Interactive evaluation loop
    print("\nEnter text to classify (or 'quit' to exit):")
    while True:
        text = input("\nText: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        # Get prediction
        result = predict_stance(text, model, tokenizer, device)
        
        # Print results
        print(f"\nPredicted stance: {result['stance']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nAll confidence scores:")
        for stance, score in result['all_scores'].items():
            print(f"{stance:>10}: {score:.2%}")

def main():
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == '__main__':
    main()
