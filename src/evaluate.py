import torch
from train import setup_model, LABELS

def load_model(model_path, model_name="roberta-base", device="cpu"):
    """Load a trained model and tokenizer"""
    model, tokenizer = setup_model(model_name)
    
    # Load the saved state dictionary
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract just the model weights from the checkpoint
    if "model_state_dict" in checkpoint:
        # If saved with full training state
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # If saved with just the model state
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model, tokenizer

def predict_stance(text, model, tokenizer, device, labels=None):
    """Predict political stance for a given text"""
    if labels is None:
        from train import LABELS
        labels = LABELS
        
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    
    # Get confidence scores for all classes
    confidence_scores = predictions[0].cpu().numpy()
    
    return {
        'stance': labels[predicted_class],
        'confidence': confidence_scores[predicted_class],
        'all_scores': {
            label: score.item()
            for label, score in zip(labels, confidence_scores)
        }
    }

def evaluate_dataset(model, tokenizer, loader, device):
    """Evaluate model on a dataset and return overall and per-label accuracies"""
    # Update for 3 labels (0: left, 1: center, 2: right)
    label_correct = {0: 0, 1: 0, 2: 0}
    label_total = {0: 0, 1: 0, 2: 0}
    
    model.eval()
    with torch.no_grad():
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
        0: "Left",
        1: "Center",
        2: "Right"
    }
    
    for label in label_total:
        if label_total[label] > 0:
            acc = label_correct[label] / label_total[label]
            label_accuracies[label_names[label]] = acc
        else:
            label_accuracies[label_names[label]] = 0.0
            
    return overall_acc, label_accuracies
