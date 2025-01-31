import os
from datetime import datetime
import argparse
from train import train_model, setup_model, setup_device
from dataset import PoliticalDataset, load_dataset
from evaluate import load_model, predict_stance
from torch.utils.data import DataLoader
from tabulate import tabulate
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Sequential Fine-tuning and Evaluation')
    parser.add_argument('--initial_model', type=str, required=True,
                       help='Path to initial trained model')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                       help='Name of the pretrained model')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training and evaluation')
    return parser.parse_args()

def evaluate_dataset(model, tokenizer, data_path, device):
    """Evaluate model on a dataset and return overall and per-label accuracies"""
    data = load_dataset(data_path)
    dataset = PoliticalDataset(data, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Track per-label metrics
    label_correct = {0: 0, 1: 0, 2: 0}
    label_total = {0: 0, 1: 0, 2: 0}
    
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        
        # Update per-label counts
        for pred, true in zip(predictions, batch['labels']):
            label = true.item()
            label_total[label] += 1
            if pred == true:
                label_correct[label] += 1
    
    # Calculate accuracies
    overall_acc = sum(label_correct.values()) / sum(label_total.values())
    
    # Calculate per-label accuracies
    label_accuracies = {}
    label_names = {
        0: "Right",
        1: "Center",
        2: "Left"
    }
    
    for label in label_total:
        if label_total[label] > 0:
            acc = label_correct[label] / label_total[label]
            label_accuracies[label_names[label]] = acc
        else:
            label_accuracies[label_names[label]] = 0.0
            
    return overall_acc, label_accuracies

def sequential_training():
    args = parse_args()
    
    # Setup - create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"sequential_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.txt")
    
    device = setup_device()
    
    # Load initial model
    print(f"Loading initial model from {args.initial_model}")
    model, tokenizer = load_model(args.initial_model, args.model_name, device)
    
    # Test datasets
    test_sets = [
        "data/processed/test1.pkl",
        "data/processed/test2.pkl",
        "data/processed/test3.pkl"
    ]
    
    # Results table
    results = []
    headers = ["Model", "Dataset", "Overall Acc", "Left", "Center", "Right"]
    
    # Initial evaluation
    print("\nEvaluating initial model...")
    for test_set in test_sets:
        dataset = PoliticalDataset(test_set, tokenizer)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        overall_acc, label_accs = evaluate_dataset(model, tokenizer, loader, device)
        results.append([
            "Initial",
            os.path.basename(test_set),
            f"{overall_acc:.4f}",
            f"{label_accs['Left']:.4f}",
            f"{label_accs['Center']:.4f}",
            f"{label_accs['Right']:.4f}"
        ])
    
    # Sequential fine-tuning
    train_data = "data/processed/train.pkl"
    print(f"\nFine-tuning on {os.path.basename(train_data)}...")
    
    # Load and split data
    df = pd.read_pickle(train_data)
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    # Save splits temporarily
    train_temp = os.path.join('data/processed', 'temp_train.pkl')
    val_temp = os.path.join('data/processed', 'temp_val.pkl')
    train_df.to_pickle(train_temp)
    val_df.to_pickle(val_temp)
    
    # Create datasets
    train_dataset = PoliticalDataset(train_temp, tokenizer)
    val_dataset = PoliticalDataset(val_temp, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Fine-tune
    train_model(
        model,
        train_loader,
        val_loader,
        device,
        model_path="final_model.pt",
        output_dir=output_dir,
        epochs=3,
        learning_rate=2e-5
    )
    
    # Clean up temporary files
    os.remove(train_temp)
    os.remove(val_temp)
    
    # Final evaluation on test sets
    print("\nEvaluating final model...")
    for test_set in test_sets:
        dataset = PoliticalDataset(test_set, tokenizer)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        overall_acc, label_accs = evaluate_dataset(model, tokenizer, loader, device)
        results.append([
            "Final",
            os.path.basename(test_set),
            f"{overall_acc:.4f}",
            f"{label_accs['Left']:.4f}",
            f"{label_accs['Center']:.4f}",
            f"{label_accs['Right']:.4f}"
        ])
    
    # Save results table
    table = tabulate(results, headers=headers, tablefmt="grid")
    with open(output_file, 'w') as f:
        f.write(table)
    print(f"\nResults saved to {output_file}")
    print("\nFinal Results:")
    print(table)

if __name__ == "__main__":
    sequential_training() 