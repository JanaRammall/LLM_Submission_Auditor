# ai_detection.py
# Uses existing chunks already stored in Chroma DB
# No duplicate chunking
# Outputs chunk percentages + final percentage

import re
from collections import Counter
from statistics import mean, pstdev


# ------------------------------------
# Basic text helpers
# ------------------------------------
def get_sentences(text):
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def get_words(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


# ------------------------------------
# Feature 1: Lexical Diversity
# ------------------------------------
def lexical_diversity(words):
    if not words:
        return 0

    return len(set(words)) / len(words)


# ------------------------------------
# Feature 2: Word Repetition
# ------------------------------------
def repetition(words):
    c = Counter(words)

    repeated = sum(v for v in c.values() if v > 2)

    return repeated / max(len(words), 1)


# ------------------------------------
# Feature 3: Sentence Statistics
# ------------------------------------
def sentence_stats(sentences):
    lengths = [len(get_words(s)) for s in sentences]

    if not lengths:
        return 0, 0

    return mean(lengths), pstdev(lengths)


# ------------------------------------
# Feature 4: Transition Overuse
# ------------------------------------
def transitions(text):
    keys = [
        "moreover", "furthermore", "therefore",
        "however", "in conclusion", "additionally",
        "thus", "overall", "consequently"
    ]

    t = text.lower()

    return sum(t.count(k) for k in keys)


# ------------------------------------
# Feature 5: Phrase Repetition
# ------------------------------------
def ngram_repetition(words, n=3):
    grams = [
        " ".join(words[i:i+n])
        for i in range(len(words)-n+1)
    ]

    if not grams:
        return 0

    c = Counter(grams)

    repeated = sum(v for v in c.values() if v > 1)

    return repeated / len(grams)


# ------------------------------------
# Feature 6: Generic AI Language
# ------------------------------------
def genericity(text):
    generic_words = [
        "many", "various", "important",
        "significant", "different",
        "modern", "society",
        "numerous", "beneficial",
        "crucial", "valuable"
    ]

    words = text.lower().split()

    count = sum(1 for w in words if w in generic_words)

    return count / max(len(words), 1)


# ------------------------------------
# Feature 7: Specificity
# Human writing often has names/numbers
# ------------------------------------
def specificity(text):
    numbers = len(re.findall(r'\d+', text))
    capitals = len(re.findall(r'\b[A-Z][a-z]+\b', text))

    return numbers + capitals


# ------------------------------------
# Score One Chunk
# ------------------------------------
def score_chunk(chunk):
    sents = get_sentences(chunk)
    words = get_words(chunk)

    diversity = lexical_diversity(words)
    rep = repetition(words)
    avg_len, std_len = sentence_stats(sents)
    trans = transitions(chunk)
    ngr = ngram_repetition(words)
    gen = genericity(chunk)
    spec = specificity(chunk)

    score = 0

    # Uniform sentence rhythm
    if std_len < 5:
        score += 20

    # Low lexical richness
    if diversity < 0.50:
        score += 15

    # Repetition
    if rep > 0.12:
        score += 15

    # Phrase reuse
    if ngr > 0.04:
        score += 20

    # Generic wording
    if gen > 0.02:
        score += 20

    # Lack of specifics
    if spec < 5:
        score += 20

    # Too smooth transitions
    if trans > 4:
        score += 10

    # Very balanced sentence length
    if 14 <= avg_len <= 24:
        score += 10

    return min(score, 100)


# ------------------------------------
# Main Detector
# ------------------------------------
def detect_ai(db):

    # Get chunks already stored in Chroma
    data = db.get()

    chunks = data["documents"]

    if not chunks:
        return "No text found."

    scores = []

    for chunk in chunks:

        score = score_chunk(chunk)

        # Semantic similarity check
        try:
            docs = db.similarity_search(chunk, k=3)

            if len(docs) >= 3:
                score += 5

        except:
            pass

        scores.append(min(score, 100))

    final_score = round(mean(scores), 2)

    # Consistency across chunks
    if len(scores) > 2:
        consistency = pstdev(scores)

        if consistency < 9:
            final_score += 10

    final_score = min(final_score, 100)

    # Final label
    if final_score >= 70:
        label = "Likely AI Generated"
    elif final_score >= 40:
        label = "Mixed / Needs Review"
    else:
        label = "Likely Human"

    # Build report
    report = []
    report.append("AI Detection Report")
    report.append("=" * 45)
    report.append(f"AI Probability: {final_score}%")
    report.append(f"Assessment : {label}")
    report.append("")

    for i, s in enumerate(scores, 1):
        report.append(f"Chunk {i}: {round(s,2)}%")

    return "\n".join(report)