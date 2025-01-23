import torch
from train import setup_model, LABELS

def load_model(model_path, model_name="roberta-base", device=None):
    """Load a trained model and tokenizer"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model(model_name)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, tokenizer

def predict_stance(text, model, tokenizer, device):
    """Predict political stance for a given text"""
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
        'stance': LABELS[predicted_class],
        'confidence': confidence_scores[predicted_class],
        'all_scores': {
            label: score.item()
            for label, score in zip(LABELS, confidence_scores)
        }
    }
