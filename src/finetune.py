import argparse
import os
from datetime import datetime
from train import train_model, setup_device
from dataset import PoliticalDataset, load_dataset
from evaluate import load_model
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Political Stance Model')
    parser.add_argument('--base_model', type=str, required=True,
                       help='Path to base model to fine-tune')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to new data for fine-tuning')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                       help='Name of the pretrained model')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs for fine-tuning')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for fine-tuning')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save fine-tuned model (default: creates timestamped dir)')
    return parser.parse_args()

def finetune():
    args = parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("outputs", f"finetuned_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = setup_device()
    
    # Load base model
    print(f"Loading base model from {args.base_model}")
    model, tokenizer = load_model(args.base_model, args.model_name, device)
    
    # Load and prepare fine-tuning data
    print(f"Loading fine-tuning data from {args.data_path}")
    data = load_dataset(args.data_path)
    print(f"Loaded {len(data)} examples for fine-tuning")
    
    # Create dataset and dataloaders
    train_dataset = PoliticalDataset(data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Fine-tune the model
    print("\nStarting fine-tuning...")
    model_path = "finetuned_model.pt"
    history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        model_path=model_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    print(f"\nFine-tuning complete! Model saved to: {os.path.join(args.output_dir, model_path)}")
    
    # Save fine-tuning configuration
    config = {
        'base_model': args.base_model,
        'fine_tuning_data': args.data_path,
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'final_accuracy': history['val_acc'][-1]
    }
    
    config_path = os.path.join(args.output_dir, 'finetune_config.txt')
    with open(config_path, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Configuration saved to: {config_path}")

if __name__ == "__main__":
    finetune() 