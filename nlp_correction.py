
# SWE015 - Classical NLP Word Correction
# Minimum Edit Distance + Smoothed Bigram Language Model

from collections import Counter
from nltk.corpus import brown
from rapidfuzz import process, fuzz, distance

# Build language model using Brown Corpus
unigrams = Counter()
bigrams = Counter()

for sentence in brown.sents():
    words = [
        word.lower()
        for word in sentence
        if word.isalpha()
    ]

    if not words:
        continue

    tokens = ["<s>"] + words + ["</s>"]

    unigrams.update(tokens)
    bigrams.update(zip(tokens[:-1], tokens[1:]))

# Keep reasonably frequent vocabulary words
vocab = sorted(
    word for word, count in unigrams.items()
    if count >= 2 and word.isalpha()
)

vocab_set = set(vocab)
V = len(vocab)


def bigram_probability(word, previous="<s>"):
    # Add-one smoothing
    return (
        bigrams[(previous.lower(), word.lower())] + 1
    ) / (
        unigrams[previous.lower()] + V
    )


def correct_word(raw_word, previous="<s>", next_word="</s>"):
    raw = raw_word.lower().strip()

    if not raw:
        return []

    # Find similar candidate words
    matches = process.extract(
        raw,
        vocab,
        scorer=fuzz.ratio,
        limit=150
    )

    candidates = []

    for word, similarity, index in matches:
        edit_distance = distance.Levenshtein.distance(
            raw, word
        )

        if edit_distance > 2:
            continue

        language_score = (
            bigram_probability(word, previous)
            * bigram_probability(next_word, word)
        )

        candidates.append(
            (word, edit_distance, language_score)
        )

    # Include the original recognized word
    if raw in vocab_set and not any(
        item[0] == raw for item in candidates
    ):
        candidates.append((
            raw,
            0,
            bigram_probability(raw, previous)
            * bigram_probability(next_word, raw)
        ))

    # Prefer smaller edit distance, then language score
    candidates.sort(
        key=lambda item: (item[1], -item[2])
    )

    return candidates[:10]


if __name__ == "__main__":
    for word in ["helo", "aple", "scholl"]:
        print(f"\nOriginal: {word}")
        print("Suggestions:", correct_word(word)[:5])
