import os
from datetime import datetime
import argparse
from train import train_model, setup_model, setup_device
from dataset import PoliticalDataset, load_dataset
from evaluate import load_model, predict_stance
from torch.utils.data import DataLoader
from tabulate import tabulate

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
    """Evaluate model on a dataset and return accuracy"""
    data = load_dataset(data_path)
    dataset = PoliticalDataset(data, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    correct = 0
    total = 0
    
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)
        
    return correct / total if total > 0 else 0

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
        "data/processed/test1.json",
        "data/processed/test2.json",
        "data/processed/test3.json"
    ]
    
    # Results table
    results = []
    headers = ["Model", "Test1", "Test2", "Test3"]
    
    # Initial evaluation
    initial_accs = []
    print("\nEvaluating initial model...")
    for test_set in test_sets:
        acc = evaluate_dataset(model, tokenizer, test_set, device)
        initial_accs.append(f"{acc:.4f}")
    results.append(["Initial"] + initial_accs)
    
    # Sequential fine-tuning and evaluation
    for i in range(len(test_sets)-1):
        train_data = test_sets[i]
        print(f"\nFine-tuning on {os.path.basename(train_data)}...")
        
        # Fine-tune
        train_dataset = PoliticalDataset(load_dataset(train_data), tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        
        train_model(
            model,
            train_loader,
            val_loader,
            device,
            model_path=f"model_after_{i+1}.pt",
            output_dir=output_dir,
            epochs=3,
            learning_rate=2e-5
        )
        
        # Evaluate on remaining test sets
        current_results = [f"After {os.path.basename(train_data)}"]
        for j, test_set in enumerate(test_sets):
            if j <= i:  # Add placeholder for already used datasets
                current_results.append("-")
            else:
                acc = evaluate_dataset(model, tokenizer, test_set, device)
                current_results.append(f"{acc:.4f}")
        results.append(current_results)
    
    # Save results table
    table = tabulate(results, headers=headers, tablefmt="grid")
    with open(output_file, 'w') as f:
        f.write(table)
    print(f"\nResults saved to {output_file}")
    print("\nFinal Results:")
    print(table)

if __name__ == "__main__":
    sequential_training() 