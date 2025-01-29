import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm.auto import tqdm
import copy

# Define our labels (5 classes)
LABELS = ['far_right', 'right', 'center', 'left', 'far_left']
NUM_LABELS = len(LABELS)

def setup_device():
    """Set and return the device (CPU/GPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def setup_model(model_name="roberta-base"):
    """Initialize the model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS
    )
    return model, tokenizer

def create_output_dir(base_dir="outputs"):
    """Create and return path to output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def validate_model(model, val_loader, device):
    """Validate the model and return accuracy and loss"""
    correct = 0
    total = 0
    total_loss = 0
    
    model.eval()
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validating", position=1, leave=False)
        for batch in val_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            predictions = torch.argmax(outputs.logits, dim=1)
            
            total_loss += loss.item()
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    return correct / total if total > 0 else 0, total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, device, model_path, output_dir=None, epochs=3, learning_rate=2e-5, 
                patience=3, min_delta=0.001):
    """
    Train the model with validation and early stopping
    
    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in validation loss to qualify as an improvement
    """
    if output_dir is None:
        output_dir = create_output_dir()
    print(f"Outputs will be saved to: {output_dir}")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_accuracy = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_path = os.path.join(output_dir, model_path)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_loss': []
    }

    epoch_pbar = tqdm(range(epochs), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Training", position=1, leave=False)
        
        for batch in batch_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=1)
            train_correct += (predictions == batch['labels']).sum().item()
            train_total += batch['labels'].size(0)
            
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation phase
        val_accuracy, val_loss = validate_model(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        history['val_loss'].append(val_loss)
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'train_acc': f'{train_accuracy:.4f}',
            'val_acc': f'{val_accuracy:.4f}',
            'val_loss': f'{val_loss:.4f}'
        })

        # Early stopping and model saving logic
        if val_loss < best_val_loss - min_delta:  # Improvement in validation loss
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            early_stopping_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, best_model_path)
            
            print(f"\nNew best model saved with validation loss: {val_loss:.4f} and accuracy: {val_accuracy:.4f}")
        else:
            early_stopping_counter += 1
            print(f"\nEarly stopping counter: {early_stopping_counter}/{patience}")
            
            if early_stopping_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break
    
    # Load best model before returning
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_accuracy']:.4f}")
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    return history

def plot_training_history(history, output_dir):
    """Plot and save training metrics"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss.png'))
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy.png'))
    plt.close()

def grid_search_lr(model, train_loader, val_loader, device, lrs=[1e-5, 2e-5, 3e-5]):
    """Perform grid search for learning rate"""
    results = {}
    
    for lr in lrs:
        print(f"\nTrying learning rate: {lr}")
        model_copy = copy.deepcopy(model)
        
        history = train_model(
            model_copy,
            train_loader,
            val_loader,
            device,
            model_path=f'model_lr_{lr}.pt',
            learning_rate=lr,
            epochs=3  # Use fewer epochs for quick testing
        )
        
        results[lr] = {
            'best_val_acc': max(history['val_acc']),
            'best_val_loss': min(history['val_loss'])
        }
    
    # Print results
    print("\nGrid Search Results:")
    for lr, metrics in results.items():
        print(f"LR: {lr:.0e}")
        print(f"Best Val Acc: {metrics['best_val_acc']:.4f}")
        print(f"Best Val Loss: {metrics['best_val_loss']:.4f}")
        
    return results
