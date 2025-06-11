import torch, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

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


from baseline import tokenized_datasets

"""
tokenized_datasets:

DatasetDict({
    train: Dataset({
        features: ['labels', 'input_ids', 'attention_mask'],
        num_rows: 25000
    })
    test: Dataset({
        features: ['labels', 'input_ids', 'attention_mask'],
        num_rows: 25000
    })
    unsupervised: Dataset({
        features: ['labels', 'input_ids', 'attention_mask'],
        num_rows: 50000
    })
})
"""

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)

from loraLinear import model

model.to(device)
print(f"INFO: Moved model to {device}")

trainable_params = [p for p in model.parameters() if p.requires_grad] # len: 24
optimizer = optim.AdamW(trainable_params, lr=learning_rate)

for epoch in range(n_epochs):
    model.train()
    print(f"\n--- Starting Epoch {epoch+1}/{n_epochs} ---")
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            num_correct += (predictions == batch["labels"]).sum().item()
            num_samples += batch["labels"].size(0)
    
    accuracy = num_correct / num_samples
    print(f"--- Epoch {epoch+1} Validation Accuracy: {accuracy:.4f} ---")

print("\nFine-tuning complete.")
torch.save(model.state_dict(), "DISTILBERT_WITH_LORA.pth")
print("Trained LoRA model saved.")