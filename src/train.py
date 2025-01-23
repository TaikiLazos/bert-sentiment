import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm.auto import tqdm

# Define our labels
LABELS = ['left', 'center', 'right']
# LABELS = ['far_left', 'left', 'center', 'right', 'far_right']
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

def train_model(model, train_loader, val_loader, device, model_path, epochs=3, learning_rate=2e-5):
    """Train the model with validation"""
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

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, model_path))
            print(f"\nNew best model saved with validation accuracy: {val_accuracy:.4f}")
    
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
