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
    """Evaluate model on a dataset and return overall and per-label accuracies"""
    data = load_dataset(data_path)
    dataset = PoliticalDataset(data, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Track per-label metrics
    label_correct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    label_total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
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
        0: "Strong Right",
        1: "Weak Right",
        2: "Center",
        3: "Weak Left",
        4: "Strong Left"
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
        "data/processed/test1.json",
        "data/processed/test2.json",
        "data/processed/test3.json"
    ]
    
    # Results table - now including per-label accuracies
    results = []
    headers = ["Model", "Dataset", "Overall Acc", "Strong Left", "Weak Left", "Center", "Weak Right", "Strong Right"]
    
    # Initial evaluation
    print("\nEvaluating initial model...")
    for test_set in test_sets:
        overall_acc, label_accs = evaluate_dataset(model, tokenizer, test_set, device)
        results.append([
            "Initial",
            os.path.basename(test_set),
            f"{overall_acc:.4f}",
            f"{label_accs['Strong Left']:.4f}",
            f"{label_accs['Weak Left']:.4f}",
            f"{label_accs['Center']:.4f}",
            f"{label_accs['Weak Right']:.4f}",
            f"{label_accs['Strong Right']:.4f}"
        ])
    
    # Sequential fine-tuning and evaluation
    for i in range(len(test_sets)-1):
        train_data = test_sets[i]
        print(f"\nFine-tuning on {os.path.basename(train_data)}...")
        
        # Fine-tune
        data = load_dataset(train_data)
        split_idx = int(len(data) * 0.8)
        train_split = data[:split_idx]
        val_split = data[split_idx:]
        
        train_dataset = PoliticalDataset(train_split, tokenizer)
        val_dataset = PoliticalDataset(val_split, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
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
        model_name = f"After {os.path.basename(train_data)}"
        for j, test_set in enumerate(test_sets):
            if j <= i:  # Skip already used datasets
                continue
            overall_acc, label_accs = evaluate_dataset(model, tokenizer, test_set, device)
            results.append([
                model_name,
                os.path.basename(test_set),
                f"{overall_acc:.4f}",
                f"{label_accs['Strong Left']:.4f}",
                f"{label_accs['Weak Left']:.4f}",
                f"{label_accs['Center']:.4f}",
                f"{label_accs['Weak Right']:.4f}",
                f"{label_accs['Strong Right']:.4f}"
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