import re
import statistics
from typing import List, Tuple, Dict

PUNCT_SET = set(".,;:!?。！？、，；：…")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r'(?<=[\.!?。！？])\s*', text)
    return [s.strip() for s in parts if s.strip()]


def sentence_feature_score(sentence: str) -> Tuple[int, Dict]:
    s = sentence.strip()
    length = len(s)
    if length == 0:
        return 50, {}

    unique_chars = len(set(s))
    unique_ratio = unique_chars / length
    punct_count = sum(ch in PUNCT_SET for ch in s)
    punct_ratio = punct_count / length
    digit_count = sum(ch.isdigit() for ch in s)
    digit_ratio = digit_count / length

    score = 50

    if length > 200:
        score += 15
    elif length > 120:
        score += 10
    elif 60 <= length <= 120:
        score += 5
    elif length < 20:
        score -= 5

    if unique_ratio < 0.35:
        score += 15
    elif unique_ratio < 0.45:
        score += 8
    elif unique_ratio > 0.7:
        score -= 5

    if punct_ratio < 0.015:
        score += 10
    elif punct_ratio < 0.03:
        score += 5
    elif punct_ratio > 0.08:
        score -= 5

    if digit_ratio > 0.15:
        score += 8
    elif digit_ratio > 0.05:
        score += 3

    ai_prob = max(5, min(95, int(round(score))))

    features = dict(
        length=length,
        unique_ratio=unique_ratio,
        punct_ratio=punct_ratio,
        digit_ratio=digit_ratio,
        ai_prob=ai_prob,
    )
    return ai_prob, features


def highlight_text(sentences: List[str], probs: List[int]) -> str:
    parts = []
    for s, p in zip(sentences, probs):
        if p >= 80:
            style = "background-color:#fee2e2;color:#991b1b;"
        elif p >= 50:
            style = "background-color:#fef3c7;color:#92400e;"
        else:
            style = "background-color:#dcfce7;color:#166534;"

        span = (
            f'<span style="{style}'
            'padding:2px 4px;border-radius:4px;margin:0 2px;display:inline-block;">'
            f'{s}</span>'
        )
        parts.append(span)

    return "".join(parts)
