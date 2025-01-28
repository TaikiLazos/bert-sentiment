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
