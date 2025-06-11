import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification
)

RANK = 4
ALPHA = 4
model_ckpt = "distilbert-base-uncased"

from loraLayer import LoRALayer

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank, alpha):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.original_layer = original_layer
        self.lora = LoRALayer(self.in_features, self.out_features, rank, alpha)
    
    def forward(self, x):
        original_output = self.original_layer(x)  # Wo*x
        lora_output = self.lora(x)                # (xA)B * scaling
        return original_output + lora_output      # Wo*x + (xA)B * scaling

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)

for param in model.parameters():
    param.requires_grad = False    # Freeze all original parameters

print("--- Injecting LoRA adapters into q_lin and v_lin layers of DISTILBERT---")
for layer in model.distilbert.transformer.layer:
    layer.attention.q_lin = LoRALinear(layer.attention.q_lin, RANK, ALPHA)
    layer.attention.v_lin = LoRALinear(layer.attention.v_lin, RANK, ALPHA)
print("INFO: LoRA Adapters INJECTED")

print("\nTrainable parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params}")
print(f"Trainable LoRA parameters: {trainable_params}")
print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.4f}%")