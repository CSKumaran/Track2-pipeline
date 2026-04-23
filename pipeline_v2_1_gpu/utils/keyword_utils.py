"""spaCy + KeyBERT keyword extraction + groundability classification."""

import logging
import os
import re
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Visual nouns that always get HIGH groundability
VISUAL_NOUNS = {
    "diagram", "chart", "graph", "table", "code", "formula", "equation",
    "image", "figure", "screenshot", "slide", "animation", "network",
    "layer", "matrix", "plot", "histogram", "bar", "pie", "scatter",
    "flowchart", "tree", "map", "photo", "picture", "video", "display",
    "screen", "interface", "button", "menu", "icon", "arrow", "box",
}

# Abstract words that get at most MEDIUM
ABSTRACT_WORDS = {
    "understanding", "importance", "approach", "however", "therefore",
    "relationship", "concept", "idea", "theory", "principle", "method",
    "way", "thing", "stuff", "aspect", "factor", "issue", "problem",
    "solution", "result", "effect", "impact", "influence", "role",
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "about", "between",
    "through", "after", "before", "during", "and", "but", "or", "nor",
    "not", "so", "yet", "both", "either", "neither", "this", "that",
    "these", "those", "it", "its", "we", "you", "he", "she", "they",
    "i", "me", "my", "our", "your", "his", "her", "their", "them",
    "also", "just", "like", "going", "really", "actually", "basically",
    "right", "well", "okay", "now", "here", "there", "then", "very",
    "much", "many", "some", "any", "all", "each", "every", "more", "most",
}

_concreteness_db = None


def load_concreteness_db(path: str = None) -> dict:
    """Load Brysbaert concreteness ratings."""
    global _concreteness_db
    if _concreteness_db is not None:
        return _concreteness_db

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "concreteness.csv")

    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Expected columns: Word, Conc.M (mean concreteness)
            word_col = "Word" if "Word" in df.columns else df.columns[0]
            conc_col = "Conc.M" if "Conc.M" in df.columns else df.columns[1]
            _concreteness_db = dict(zip(
                df[word_col].str.lower().str.strip(),
                pd.to_numeric(df[conc_col], errors="coerce")
            ))
            logger.info("Loaded concreteness DB: %d entries", len(_concreteness_db))
        except Exception as e:
            logger.warning("Failed to load concreteness DB: %s", e)
            _concreteness_db = {}
    else:
        logger.warning("Concreteness DB not found at %s, using defaults", path)
        _concreteness_db = {}

    return _concreteness_db


def extract_keywords_spacy(text: str) -> list:
    """Extract keywords using spaCy (noun chunks + entities + standalone nouns)."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        return []

    doc = nlp(text)
    keywords = set()

    # Noun chunks
    for chunk in doc.noun_chunks:
        kw = chunk.text.strip().lower()
        # Remove leading determiners/pronouns
        kw = re.sub(r'^(the|a|an|this|that|these|those|some|any|my|your|his|her|its|our|their)\s+', '', kw)
        if len(kw) >= 3 and kw not in STOP_WORDS:
            keywords.add(kw)

    # Named entities
    keep_types = {"WORK_OF_ART", "LAW", "LANGUAGE", "NORP", "ORG", "PRODUCT"}
    skip_types = {"PERSON", "DATE", "MONEY", "CARDINAL", "ORDINAL", "TIME", "PERCENT"}
    for ent in doc.ents:
        if ent.label_ in keep_types:
            kw = ent.text.strip().lower()
            if len(kw) >= 3:
                keywords.add(kw)

    # Standalone nouns not in any chunk
    chunk_tokens = set()
    for chunk in doc.noun_chunks:
        for tok in chunk:
            chunk_tokens.add(tok.i)

    for tok in doc:
        if tok.pos_ == "NOUN" and tok.i not in chunk_tokens:
            kw = tok.text.strip().lower()
            if len(kw) >= 3 and kw not in STOP_WORDS:
                keywords.add(kw)

    return list(keywords)


def extract_keywords_keybert(text: str, model_name: str = "BAAI/bge-large-en-v1.5",
                             top_n: int = 5) -> list:
    """Extract keywords using KeyBERT."""
    try:
        from keybert import KeyBERT
        kw_model = KeyBERT(model=model_name)
        keywords = kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 3),
            stop_words="english", top_n=top_n,
            use_mmr=True, diversity=0.5
        )
        return [kw for kw, score in keywords if len(kw) >= 3]
    except Exception as e:
        logger.warning("KeyBERT extraction failed: %s", e)
        return []


def merge_keywords(spacy_kws: list, keybert_kws: list, min_length: int = 3) -> list:
    """Merge and deduplicate keywords from both methods."""
    all_kws = set()
    for kw in spacy_kws + keybert_kws:
        kw = kw.strip().lower()
        if len(kw) >= min_length and kw not in STOP_WORDS:
            all_kws.add(kw)

    # Deduplicate: prefer longer phrase over single word
    result = list(all_kws)
    to_remove = set()
    for i, kw1 in enumerate(result):
        for j, kw2 in enumerate(result):
            if i != j and kw1 in kw2 and len(kw1) < len(kw2):
                to_remove.add(kw1)

    return [kw for kw in result if kw not in to_remove]


def classify_groundability(keyword: str, concreteness_db: dict = None) -> str:
    """Classify keyword groundability: HIGH, MEDIUM, LOW."""
    kw_lower = keyword.lower().strip()

    # Override: visual nouns → HIGH
    for word in kw_lower.split():
        if word in VISUAL_NOUNS:
            return "HIGH"

    # Override: abstract words → at most MEDIUM (but check concreteness first)
    is_single = len(kw_lower.split()) == 1

    # Concreteness DB lookup
    if concreteness_db:
        # For multi-word, average concreteness
        words = kw_lower.split()
        scores = []
        for w in words:
            if w in concreteness_db:
                scores.append(concreteness_db[w])

        if scores:
            avg_conc = np.mean(scores)
            if avg_conc >= 4.0:
                return "HIGH"
            elif avg_conc >= 2.5:
                return "MEDIUM"
            else:
                return "LOW"

    # Multi-word technical term → at least MEDIUM
    if len(kw_lower.split()) > 1:
        return "MEDIUM"

    # Single abstract word → LOW
    if is_single and kw_lower in ABSTRACT_WORDS:
        return "LOW"

    # Default
    return "MEDIUM"
