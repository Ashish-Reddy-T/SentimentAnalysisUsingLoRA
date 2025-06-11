import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
from torch.utils.data import DataLoader

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

"""
--- Model ---
"""

model_ckpt = "distilbert-base-uncased"

print(f"--- Loading pre-trained model and tokenizer: {model_ckpt.upper()} ---")

tok = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
model.to(device)
print(f"Model moved to {device}")


"""
--- Data Prep ---
"""

print("\n--- Loading and preparing IMDB dataset ---")
imdb_dataset = load_dataset("imdb")
"""
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['text', 'label'],
        num_rows: 50000
    })
})
"""

def tokenize_fn(examples):
    return tok(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = imdb_dataset.map(tokenize_fn, batched=True)

tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")


if __name__ == '__main__':

    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000)) # Select random 1000 test datasets
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8) # Convert them into 8 batches --> 125 ['labels', 'token_ids', 'attention_mask'] examples in each batch

    print("\n--- Evaluating baseline model performance ---")
    model.eval() 
    num_correct = 0
    num_samples = 0

    with torch.no_grad(): # Disable gradient calculation for inference (No backprop)
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch) # Forward pass
            logits = outputs.logits  # Logits
            
            predictions = torch.argmax(logits, dim=-1) # Highest logit score
            
            # Compare predictions to true labels
            num_correct += (predictions == batch["labels"]).sum().item()
            num_samples += batch["labels"].size(0)

    accuracy = num_correct / num_samples
    print(f"Baseline Accuracy on 1000 samples: {accuracy:.4f}") # Around 0.4880 --> 48% accurate (For 1000 testing examples) [As it plays the game of guessing, it always is around the 50% mark as the model isn't still trained and you can expect the output to be always positive or always negative]