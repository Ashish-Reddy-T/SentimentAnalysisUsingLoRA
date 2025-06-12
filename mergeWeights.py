import torch
from transformers import AutoModelForSequenceClassification
from loraLinear import LoRALinear

MODEL_CKPT   = "distilbert-base-uncased"
RANK         = 4
ALPHA        = 4
DEVICE       = "cpu"                  # fine for Spaces; merge is fast

# Re-create the LoRA-injected architecture
lora_model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT)
for blk in lora_model.distilbert.transformer.layer:
    blk.attention.q_lin = LoRALinear(blk.attention.q_lin, RANK, ALPHA)
    blk.attention.v_lin = LoRALinear(blk.attention.v_lin, RANK, ALPHA)

lora_model.load_state_dict(torch.load("DISTILBERT_WITH_LORA.pth", map_location=DEVICE))
lora_model.eval()

# Collapse each adapter:  W ← W + (B @ A)·scale
for blk in lora_model.distilbert.transformer.layer:
    for name in ("q_lin", "v_lin"):
        wrap = getattr(blk.attention, name)
        with torch.no_grad():
            base_W = wrap.original_layer.weight        # (out, in)
            A      = wrap.lora.loraA.weight             # (rank, in)
            B      = wrap.lora.loraB.weight             # (out, rank)
            base_W += (B @ A) * wrap.lora.scaling       # in-place update

# Copy the merged weights into a *plain* DistilBERT (no wrappers)
plain_model = AutoModelForSequenceClassification.from_pretrained(MODEL_CKPT)
with torch.no_grad():
    for i in range(6):
        plain_blk = plain_model.distilbert.transformer.layer[i]
        lora_blk  = lora_model.distilbert.transformer.layer[i]

        for lin in ("q_lin", "v_lin"):
            pl = getattr(plain_blk.attention, lin)
            lr = getattr(lora_blk.attention, lin).original_layer
            pl.weight.copy_(lr.weight)
            pl.bias.copy_(lr.bias)

    # classification head
    plain_model.pre_classifier.weight.copy_(lora_model.pre_classifier.weight)
    plain_model.pre_classifier.bias.copy_(lora_model.pre_classifier.bias)
    plain_model.classifier.weight.copy_(lora_model.classifier.weight)
    plain_model.classifier.bias.copy_(lora_model.classifier.bias)

# Save
torch.save(plain_model.state_dict(), "DISTILBERT_MERGED.pth")
print("✅  Merged weights saved to  DISTILBERT_MERGED.pth")