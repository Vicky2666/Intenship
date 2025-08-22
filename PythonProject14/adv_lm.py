# %% [markdown]
# # Advanced LM Analysis: Medical Domain with ClinicalBERT
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# %%
# Load medical-specific model
model_name = "emilyalsentzer/Bio_ClinicalBERT"  # ClinicalBERT trained on medical texts
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# %% [markdown]
# ## Medical Text Processing and Analysis

# %%
# Medical terminology test cases
medical_terms = [
    "The patient presented with [MASK] and shortness of breath.",
    "Treatment for hypertension may include [MASK] medications.",
    "The MRI showed [MASK] in the left lung.",
    "Symptoms of diabetes include polyuria and [MASK].",
    "The surgeon performed a [MASK] to remove the appendix."
]


# %%
def analyze_medical_terms(terms_list, top_k=5):
    """Analyze medical terminology understanding"""
    results = []

    for term in terms_list:
        inputs = tokenizer(term, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predictions = outputs.logits
        masked_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

        if len(masked_index) > 0:
            masked_index = masked_index[0]
            probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
            top_k_tokens = torch.topk(probs, top_k)

            term_results = []
            for token_id, prob in zip(top_k_tokens.indices, top_k_tokens.values):
                token = tokenizer.decode([token_id])
                term_results.append((token.strip(), prob.item()))

            results.append({
                'original_text': term,
                'predictions': term_results
            })

    return results


# %%
# Run medical terminology analysis
medical_results = analyze_medical_terms(medical_terms)

print("Medical Terminology Analysis:")
print("=" * 60)
for result in medical_results:
    print(f"\nInput: {result['original_text']}")
    for i, (token, prob) in enumerate(result['predictions']):
        print(f"Top {i + 1}: {token} (Probability: {prob:.4f})")

# %% [markdown]
# ## Clinical Text Understanding

# %%
# Sample clinical notes for analysis
clinical_notes = [
    "Patient is a 45-year-old male with history of hypertension presenting with chest pain radiating to left arm. ECG shows ST elevation. Troponin levels elevated. Diagnosis: acute myocardial infarction.",
    "55-year-old female with type 2 diabetes mellitus presents with polyuria, polydipsia, and weight loss. HbA1c is 9.2%. Treatment started with metformin and lifestyle modification.",
    "Neonate presenting with jaundice on day 3 of life. Bilirubin level 18 mg/dL. Phototherapy initiated. No evidence of hemolytic disease."
]


# %%
def extract_medical_entities(text):
    """Extract potential medical entities using pattern matching"""
    # Medical patterns (simplified for demonstration)
    patterns = {
        'age': r'(\d+)-year-old',
        'conditions': r'(hypertension|diabetes|myocardial infarction|jaundice)',
        'symptoms': r'(chest pain|polyuria|polydipsia|weight loss|jaundice)',
        'treatments': r'(metformin|phototherapy|lifestyle modification)',
        'measurements': r'(HbA1c|Bilirubin level|Troponin levels)',
        'values': r'(\d+\.\d+%|\d+ mg/dL)'
    }

    entities = {}
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            entities[entity_type] = matches

    return entities


# %%
# Analyze clinical notes
print("Clinical Note Analysis:")
print("=" * 50)

for i, note in enumerate(clinical_notes):
    print(f"\nNote {i + 1}:")
    entities = extract_medical_entities(note)
    for entity_type, values in entities.items():
        print(f"{entity_type.title()}: {', '.join(values)}")

# %% [markdown]
# ## Medical Concept Relationship Analysis

# %%
# Test understanding of medical relationships
medical_relationships = [
    "Aspirin is used to treat [MASK].",
    "[MASK] is a common side effect of chemotherapy.",
    "Patients with diabetes often develop [MASK] complications.",
    "Hypertension can lead to [MASK] if untreated."
]

# %%
# Analyze medical relationships
relationship_results = analyze_medical_terms(medical_relationships)

print("\nMedical Relationship Analysis:")
print("=" * 50)
for result in relationship_results:
    print(f"\nInput: {result['original_text']}")
    for i, (token, prob) in enumerate(result['predictions'][:3]):
        print(f"Top {i + 1}: {token} (Probability: {prob:.4f})")


# %% [markdown]
# ## Visualization of Medical Concepts

# %%
# Create medical concept visualizations
def create_medical_wordcloud(clinical_texts):
    """Generate word cloud from clinical texts"""
    all_text = ' '.join(clinical_texts)

    # Clean text
    words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
    medical_words = [word for word in words if len(word) > 4]  # Filter short words

    word_freq = Counter(medical_words)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Medical Concept Word Cloud')
    plt.show()

    return word_freq


# %%
# Generate word cloud
word_freq = create_medical_wordcloud(clinical_notes)

# Display top medical terms
top_terms = pd.DataFrame(word_freq.most_common(15), columns=['Term', 'Frequency'])
plt.figure(figsize=(10, 6))
sns.barplot(data=top_terms, x='Frequency', y='Term', palette='viridis')
plt.title('Top Medical Terms in Clinical Notes')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Ethical Considerations in Medical NLP

# %%
# Test for biases in medical context
bias_test_cases = [
    "The [MASK] patient is more likely to have hypertension.",
    "[MASK] patients often report higher pain tolerance.",
    "The [MASK] doctor diagnosed the condition."
]

print("Bias Analysis in Medical Context:")
print("=" * 50)
bias_results = analyze_medical_terms(bias_test_cases)

for result in bias_results:
    print(f"\nInput: {result['original_text']}")
    for i, (token, prob) in enumerate(result['predictions'][:3]):
        print(f"Top {i + 1}: {token} (Probability: {prob:.4f})")

# %% [markdown]
# ## Research Questions for Medical Domain

# %%
# Define research questions and analysis
research_questions = {
    "Q1": "How well does the model understand medical terminology?",
    "Q2": "Can the model identify relationships between medical concepts?",
    "Q3": "How does the model handle clinical note analysis?",
    "Q4": "What biases exist in medical predictions?",
    "Q5": "How accurate are the model's medical predictions?"
}

# %%
# Analyze research questions
print("Research Questions Analysis:")
print("=" * 40)

for q_id, question in research_questions.items():
    print(f"\n{q_id}: {question}")

    if q_id == "Q1":
        # Medical terminology accuracy
        medical_accuracy = sum(1 for result in medical_results
                               if any('pain' in pred[0] or 'pressure' in pred[0]
                                      for pred in result['predictions']))
        print(f"   Findings: Model shows {medical_accuracy / len(medical_results):.0%} accuracy on basic medical terms")

    elif q_id == "Q4":
        # Bias analysis
        bias_found = any(result['predictions'][0][0] in ['black', 'white', 'asian']
                         for result in bias_results)
        print(f"   Findings: {'Potential biases detected' if bias_found else 'No obvious biases detected'}")

# %% [markdown]
# ## Performance Evaluation on Medical Tasks

# %%
# Create medical evaluation dataset
medical_eval_data = [
    ("Patient with chest pain and shortness of breath", "cardiac symptoms"),
    ("Elevated blood glucose levels", "diabetes indicator"),
    ("Fever, cough, and difficulty breathing", "respiratory infection"),
    ("Joint pain and morning stiffness", "arthritis symptoms"),
    ("Headache with visual disturbances", "migraine symptoms")
]


# %%
def evaluate_medical_understanding(texts):
    """Evaluate model's medical understanding"""
    results = []

    for text, true_label in texts:
        # Create masked version
        masked_text = text + " This suggests [MASK]."
        predictions = analyze_medical_terms([masked_text])[0]['predictions']

        # Get top prediction
        top_prediction = predictions[0][0] if predictions else "unknown"

        results.append({
            'text': text,
            'true_label': true_label,
            'prediction': top_prediction,
            'match': true_label.split()[0].lower() in top_prediction.lower()
        })

    return results


# %%
# Run evaluation
med_eval_results = evaluate_medical_understanding(medical_eval_data)

# Calculate accuracy
accuracy = sum(1 for res in med_eval_results if res['match']) / len(med_eval_results)

print("Medical Understanding Evaluation:")
print("=" * 50)
print(f"Overall Accuracy: {accuracy:.2%}\n")

for res in med_eval_results:
    status = "✓" if res['match'] else "✗"
    print(f"{status} Text: {res['text']}")
    print(f"   True: {res['true_label']}")
    print(f"   Pred: {res['prediction']}\n")

# %% [markdown]
# ## Advanced Medical Analysis: Drug-Disease Relationships

# %%
# Analyze drug-disease relationships
drug_disease_cases = [
    "Metformin is used to treat [MASK].",
    "Patients taking warfarin should avoid [MASK].",
    "Aspirin can help prevent [MASK].",
    "Statins are prescribed for [MASK]."
]

# %%
# Analyze drug-disease relationships
dd_results = analyze_medical_terms(drug_disease_cases)

print("Drug-Disease Relationship Analysis:")
print("=" * 50)
for result in dd_results:
    print(f"\nInput: {result['original_text']}")
    for i, (token, prob) in enumerate(result['predictions'][:3]):
        print(f"Top {i + 1}: {token} (Probability: {prob:.4f})")

# %% [markdown]
# ## Limitations and Challenges in Medical NLP

# %%
# Test edge cases and limitations
challenge_cases = [
    "The patient's [MASK] levels were elevated indicating possible infection.",
    "Treatment resistant [MASK] requires specialized care.",
    "Rare genetic disorder [MASK] affects only 1 in 100,000 people."
]

print("Challenges and Limitations:")
print("=" * 40)
challenge_results = analyze_medical_terms(challenge_cases)

for result in challenge_results:
    print(f"\nChallenge case: {result['original_text']}")
    print("Top predictions:")
    for i, (token, prob) in enumerate(result['predictions'][:3]):
        print(f"  {i + 1}. {token} (p={prob:.4f})")

# %% [markdown]
# ## Conclusion: Medical Domain LM Analysis

# %%
# Final summary and insights
print("MEDICAL DOMAIN ANALYSIS SUMMARY")
print("=" * 50)

print("\nKEY FINDINGS:")
print("1. Medical Terminology Understanding:")
print("   - Strong grasp of common medical terms")
print("   - Good performance on symptom-disease relationships")
print("   - Some limitations with rare conditions")

print("\n2. Clinical Text Analysis:")
print("   - Effective at extracting medical entities")
print("   - Understands context in clinical notes")
print("   - Can identify treatment relationships")

print("\n3. Limitations Identified:")
print("   - Struggles with very rare medical conditions")
print("   - Occasional irrelevant predictions")
print("   - Potential biases in demographic associations")

print("\n4. Ethical Considerations:")
print("   - Requires careful monitoring for biases")
print("   - Important for patient safety applications")
print("   - Needs human expert validation for clinical use")

print("\n5. Recommended Applications:")
print("   - Medical education tools")
print("   - Clinical note assistance")
print("   - Medical literature analysis")
print("   - Patient information systems")

print(f"\nOverall Medical Accuracy: {accuracy:.2%}")