"""Thai text normalisation and CER / WER for the ASR intelligibility metric.

normalize_thai() is applied identically to the reference sentence and the ASR
hypothesis:

1. Unicode NFKC (full-width forms, compatibility characters).
2. pythainlp.util.normalize: canonical Thai mark order, duplicate marks removed,
   SARA AM recomposed (NFKC splits it into NIKHAHIT + SARA AA), and so on.
3. Lower-case (Latin).
4. Digits spelled out in Thai words. Thai digits are first mapped to Arabic,
   thousands separators are dropped, an integer becomes
   pythainlp.util.num_to_thaiword, and a decimal becomes integer + "จุด" +
   one word per digit. "๒๕๖๔", "2564" and "2,564" all become
   "สองพันห้าร้อยหกสิบสี่".
5. Punctuation, symbols, whitespace and control characters (Unicode categories
   P*, S*, Z*, C*) removed. Thai letters, vowels and tone marks, MAI YAMOK (ๆ)
   and PAIYANNOI (ฯ) are kept.

CER: Levenshtein distance over the characters of the normalised strings,
divided by the reference length. WER: the same over words from one fixed
tokenizer, PyThaiNLP newmm (keep_whitespace=False), applied to the normalised
(space-free) strings. Both are per clip; corpus-level rates are the sums of
edits over the sums of reference lengths (scripts/summarize_cv_eval.py).
"""

import re
import unicodedata

import pythainlp
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize as pythainlp_normalize
from pythainlp.util import num_to_thaiword

NORMALIZATION_VERSION = "thai_text-v1"
WORD_TOKENIZER = "newmm"
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
DIGIT_WORDS = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def _spell_number(match):
    text = match.group(0).replace(",", "")
    integer, _, decimals = text.partition(".")
    words = num_to_thaiword(int(integer))
    if decimals:
        words += "จุด" + "".join(DIGIT_WORDS[int(d)] for d in decimals)
    return words


def normalize_thai(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = pythainlp_normalize(text)
    text = text.lower().translate(THAI_DIGITS)
    text = NUMBER.sub(_spell_number, text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZC")


def tokenize_words(normalized):
    return [w for w in word_tokenize(normalized, engine=WORD_TOKENIZER, keep_whitespace=False) if w.strip()]


def edit_distance(reference, hypothesis):
    """Levenshtein distance between two sequences (substitution = insertion = deletion = 1)."""
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def error_rates(reference, hypothesis):
    """Per-clip CER / WER with their edit counts and reference lengths."""
    ref, hyp = normalize_thai(reference), normalize_thai(hypothesis)
    ref_words, hyp_words = tokenize_words(ref), tokenize_words(hyp)
    if not ref:
        raise ValueError(f"empty reference after normalisation: {reference!r}")
    char_edits = edit_distance(ref, hyp)
    word_edits = edit_distance(ref_words, hyp_words)
    return {"cer": char_edits / len(ref), "wer": word_edits / len(ref_words),
            "cer_edits": char_edits, "cer_ref_len": len(ref),
            "wer_edits": word_edits, "wer_ref_len": len(ref_words),
            "ref_norm": ref, "hyp_norm": hyp}


def text_settings():
    return {"normalization": NORMALIZATION_VERSION, "word_tokenizer": WORD_TOKENIZER,
            "pythainlp": pythainlp.__version__}
