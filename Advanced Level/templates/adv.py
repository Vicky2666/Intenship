# %% [markdown]
# # Advanced LM Analysis: Medical Domain with ClinicalBERT + Gradio Demo

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import gradio as gr

# Load ClinicalBERT
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function for Gradio demo
def medical_mask_predict(text, top_k=5):
    """Predict masked token(s) in medical text"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits
    masked_indices = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    results = []
    for idx in masked_indices:
        probs = torch.nn.functional.softmax(predictions[0, idx], dim=-1)
        top_tokens = torch.topk(probs, top_k)

        for token_id, prob in zip(top_tokens.indices, top_tokens.values):
            token = tokenizer.decode([token_id]).strip()
            results.append(f"{token} ({prob:.4f})")

    return "\n".join(results) if results else "⚠️ No [MASK] token found in input."

# Gradio interface
demo = gr.Interface(
    fn=medical_mask_predict,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter a medical sentence with [MASK] ..."),
        gr.Slider(1, 10, value=5, step=1, label="Top K Predictions")
    ],
    outputs="text",
    title="🩺 ClinicalBERT Medical LM Demo",
    description="Type a clinical sentence with [MASK] and see ClinicalBERT predictions."
)

# Run demo
if __name__ == "__main__":
    demo.launch()
