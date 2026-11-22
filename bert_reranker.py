
# SWE015 - Quantized BERT Word Reranking

import os
import torch
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM
)

from nlp_correction import correct_word

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BERT_WEIGHTS = os.path.join(
    BASE_DIR,
    "quantized_models",
    "bert_int8_weights.pt"
)

MODEL_NAME = "google-bert/bert-base-uncased"

# Load tokenizer and model architecture
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

config = BertConfig.from_pretrained(MODEL_NAME)
bert_model = BertForMaskedLM(config).cpu().eval()

# Recreate the same dynamic INT8 architecture
bert_model = torch.ao.quantization.quantize_dynamic(
    bert_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Load our quantized weights from Colab
state_dict = torch.load(
    BERT_WEIGHTS,
    map_location="cpu",
    weights_only=True
)

bert_model.load_state_dict(state_dict)
bert_model.eval()

print("Quantized BERT loaded successfully!")


def bert_correct_word(raw_word, sentence):
    candidates = correct_word(raw_word)

    if not candidates:
        return raw_word, []

    # Example:
    # I said [MASK] to my friend.
    if "[MASK]" not in sentence:
        raise ValueError(
            "Sentence must contain [MASK]."
        )

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    mask_positions = (
        inputs["input_ids"][0] == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)[0]

    if len(mask_positions) != 1:
        raise ValueError("Use exactly one [MASK].")

    with torch.inference_mode():
        outputs = bert_model(**inputs)

    probabilities = torch.softmax(
        outputs.logits[0, mask_positions[0]],
        dim=-1
    )

    ranked = []

    for word, edit_distance, ngram_score in candidates:
        token_ids = tokenizer.encode(
            word,
            add_special_tokens=False
        )

        # BERT mask predicts one token
        if len(token_ids) != 1:
            continue

        score = float(probabilities[token_ids[0]].item())

        ranked.append({
            "word": word,
            "bert_score": score,
            "edit_distance": edit_distance,
            "ngram_score": ngram_score
        })

    ranked.sort(
        key=lambda item: item["bert_score"],
        reverse=True
    )

    if not ranked:
        return raw_word, []

    return ranked[0]["word"], ranked


if __name__ == "__main__":
    corrected, ranking = bert_correct_word(
        "helo",
        "I said [MASK] to my friend."
    )

    print("Raw word: helo")
    print("Corrected word:", corrected)
    print("Top candidates:", ranking[:3])
