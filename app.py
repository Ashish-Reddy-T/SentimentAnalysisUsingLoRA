import torch, torch.nn.functional as F
import gradio as gr

model_ckpt = "distilbert-base-uncased"
batch_size = 16
n_epochs = 3
learning_rate = 1e-4
RANK = 4
ALPHA = 4


"""
---- Device ----
"""

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA (GPU)")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
    print("Using MPS (Apple Silicon GPU)")
else:
    device = torch.device('cpu')
    print("Using device's CPU")


from baseline import model, tok


print("--- Loading fine-tuned LoRA weights ---")
model.load_state_dict(torch.load('DISTILBERT_WITH_LORA.pth', map_location=device))
model.to(device)
model.eval()

print("Model ready.")

def predict_sentiment(text):
    inputs = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=-1)
    
    labels = model.config.id2label
    confidences = {labels[i]: p.item() for i, p in enumerate(probs[0])}
    
    return confidences

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, label="Movie Review", placeholder="Enter a movie review here..."),
    outputs=gr.Label(num_top_classes=2, label="Sentiment"),
    title="Sentiment Analysis with a LoRA-tuned DistilBERT",
    description="This is a demo for a DistilBERT model that was fine-tuned for sentiment analysis using a custom LoRA implementation from scratch. Enter a movie review to see the model's prediction.",
    examples=[
        ["This movie was an absolute masterpiece. The acting was superb and the story was gripping!"],
        ["I would not recommend this film to my worst enemy. It was a complete waste of time."],
        ["The plot was a bit predictable, but the special effects were stunning."],
        ["I'm not sure how I feel about this movie. It had some good moments but was also very slow."]
    ]
)

iface.launch()