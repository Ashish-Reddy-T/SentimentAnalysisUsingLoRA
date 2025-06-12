import torch, torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_CKPT = "distilbert-base-uncased"
DEVICE     = "cpu"     # HF Spaces default

print("--- Loading tokenizer & base model ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT)

print("--- Loading merged fine-tuned weights ---")
model.load_state_dict(torch.load("DISTILBERT_MERGED.pth", map_location=DEVICE))
model.to(DEVICE).eval()

# nice label names for IMDB
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {v: k for k, v in model.config.id2label.items()}

def predict(text):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        probs = F.softmax(model(**tokens).logits, dim=-1)[0]
    return {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, label="Movie Review"),
    outputs=gr.Label(num_top_classes=2, label="Sentiment"),
    title="Sentiment Analysis (LoRA-merged DistilBERT)",
    description=(
        "DistilBERT fine-tuned on IMDB with a custom LoRA adapter. "
        "Adapters have been merged so the model runs with no extra parameters."
    ),
    examples=[
        ["An absolute masterpiece with brilliant acting!"],
        ["Total waste of two hours."],
        ["Predictable plot but gorgeous visuals."]
    ]
)

if __name__ == "__main__":
    demo.launch()