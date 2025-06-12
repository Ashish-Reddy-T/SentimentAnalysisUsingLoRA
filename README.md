# LoRA (Low-Rank Adaptation) from Scratch for Sentiment Analysis

## Details

- Used `distilbert-base-uncased` as the base model.
- The `LoRA` pipeline was entirely build from scratch replicating the architecture included in their original  __[paper](https://arxiv.org/abs/2106.09685)__.
- Tested on the `imdb` dataset from `huggingface` for sentiment analysis:
    - Validation accuracy on the original base model: `~50%`
    - Validation accuracy after applying LoRA: `91.4%` (3 epochs)


---

## Running it

You can run your own review statements from my deployed huggingface's [SPACE](https://huggingface.co/spaces/Ashish-R/LoRAFromScratchSentimentAnalysis) 💖

- It shows the percentage for the model thinks a statement is `negative` and `positive`. 
- Any modifications/suggestions are always welcome!

---