"""spaCy + KeyBERT keyword extraction + groundability classification [v2.2]."""

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

# [v2.2] Vague/auxiliary single words → LOW when no concreteness DB
# These are words that spaCy tags as NOUN but have no visual referent
VAGUE_NOUNS = {
    "one", "order", "point", "time", "part", "bit", "lot", "kind", "sort",
    "case", "example", "step", "move", "try", "start", "end", "turn",
    "need", "use", "work", "look", "set", "run", "call", "let", "hold",
    "think", "imagine", "mean", "mind", "sense", "deal", "change", "chance",
    "focus", "process", "quality", "option", "choice", "terms", "basis",
    "essence", "manner", "nature", "respect", "regard", "behalf",
}

# [v2.2] Domain-specific terms that ARE visual in educational CS/AI context
# These override ABSTRACT_WORDS and get at least MEDIUM
DOMAIN_VISUAL_TERMS = {
    # Algorithms & methods
    "hill climbing", "simulated annealing", "local search", "gradient descent",
    "genetic algorithm", "neural network", "decision tree", "random forest",
    "backpropagation", "clustering", "classification", "regression",
    # Data structures
    "stack", "queue", "linked list", "binary tree", "hash table", "heap",
    # Math/optimization
    "function", "objective function", "cost function", "fitness function",
    "maximum", "minimum", "optimum", "global maximum", "local maximum",
    "plateau", "ridge", "valley", "landscape", "search space",
    "temperature", "cooling", "convergence",
    # Problems
    "traveling salesman", "knapsack", "scheduling", "routing",
    # General visual
    "state", "neighbor", "path", "node", "edge", "weight",
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
_spacy_nlp = None
_keybert_model = None


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


def _get_spacy_nlp():
    """Lazy-load spaCy model (cached)."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy en_core_web_sm")
    return _spacy_nlp


def _get_keybert_model(model_name: str):
    """Lazy-load KeyBERT model (cached)."""
    global _keybert_model
    if _keybert_model is None:
        from keybert import KeyBERT
        _keybert_model = KeyBERT(model=model_name)
        logger.info("Loaded KeyBERT with %s", model_name)
    return _keybert_model


def extract_keywords_spacy(text: str) -> list:
    """Extract keywords using spaCy (noun chunks + entities + standalone nouns)."""
    try:
        nlp = _get_spacy_nlp()
    except Exception:
        return []

    doc = nlp(text)
    keywords = set()

    # Noun chunks
    for chunk in doc.noun_chunks:
        kw = chunk.text.strip().lower()
        kw = re.sub(
            r'^(the|a|an|this|that|these|those|some|any|my|your|his|her|its|our|their)\s+',
            '', kw
        )
        if len(kw) >= 3 and kw not in STOP_WORDS:
            keywords.add(kw)

    # Named entities
    keep_types = {"WORK_OF_ART", "LAW", "LANGUAGE", "NORP", "ORG", "PRODUCT"}
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
        kw_model = _get_keybert_model(model_name)
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
    """Merge and deduplicate keywords from both methods.

    [v2.2] Improved deduplication:
    1. Remove exact substrings (original logic)
    2. Remove phrases that share >50% words with a longer phrase
    3. Remove single vague/stop words that snuck through
    """
    all_kws = set()
    for kw in spacy_kws + keybert_kws:
        kw = kw.strip().lower()
        if len(kw) >= min_length and kw not in STOP_WORDS:
            all_kws.add(kw)

    result = list(all_kws)
    to_remove = set()

    # Pass 1: Remove exact substrings (prefer longer phrase)
    for i, kw1 in enumerate(result):
        for j, kw2 in enumerate(result):
            if i != j and kw1 in kw2 and len(kw1) < len(kw2):
                to_remove.add(kw1)

    result = [kw for kw in result if kw not in to_remove]
    to_remove = set()

    # Pass 2: Remove phrases with high word overlap (>50% shared words)
    for i, kw1 in enumerate(result):
        words1 = set(kw1.split())
        for j, kw2 in enumerate(result):
            if i >= j:
                continue
            words2 = set(kw2.split())
            overlap = words1 & words2
            # If shorter phrase shares >50% of its words with longer, remove shorter
            if len(words1) < len(words2):
                if len(overlap) > len(words1) * 0.5:
                    to_remove.add(kw1)
            elif len(words2) < len(words1):
                if len(overlap) > len(words2) * 0.5:
                    to_remove.add(kw2)

    return [kw for kw in result if kw not in to_remove]


def classify_groundability(keyword: str, concreteness_db: dict = None) -> str:
    """Classify keyword groundability: HIGH, MEDIUM, LOW.

    [v2.2] Stricter without concreteness DB:
    - VISUAL_NOUNS → HIGH
    - DOMAIN_VISUAL_TERMS → MEDIUM (even if abstract words present)
    - ABSTRACT_WORDS (single) → LOW
    - VAGUE_NOUNS (single) → LOW
    - Multi-word with all vague/abstract words → LOW
    - Multi-word with at least one content word → MEDIUM
    """
    kw_lower = keyword.lower().strip()
    words = kw_lower.split()
    is_single = len(words) == 1

    # Override: visual nouns → HIGH
    for word in words:
        if word in VISUAL_NOUNS:
            return "HIGH"

    # Override: domain-specific visual terms → at least MEDIUM
    if kw_lower in DOMAIN_VISUAL_TERMS:
        return "MEDIUM"
    # Check partial domain term matches (e.g., "local search algorithms" contains "local search")
    for term in DOMAIN_VISUAL_TERMS:
        if term in kw_lower or kw_lower in term:
            return "MEDIUM"

    # Concreteness DB lookup (when available)
    if concreteness_db:
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

    # === No concreteness DB: stricter rules ===

    # Single word checks
    if is_single:
        if kw_lower in ABSTRACT_WORDS:
            return "LOW"
        if kw_lower in VAGUE_NOUNS:
            return "LOW"
        # Single word, not in any special list, no concreteness DB
        # Default to LOW for single words (too ambiguous without concreteness data)
        if len(kw_lower) <= 4:
            return "LOW"
        return "MEDIUM"

    # Multi-word: check if ALL words are vague/abstract/stop
    non_content_words = VAGUE_NOUNS | ABSTRACT_WORDS | STOP_WORDS
    content_words = [w for w in words if w not in non_content_words and len(w) >= 3]

    if not content_words:
        # All words are vague/abstract → LOW
        return "LOW"

    # Multi-word with at least one content word → MEDIUM
    return "MEDIUM"


def unload_keyword_models():
    """Free keyword extraction model resources."""
    global _spacy_nlp, _keybert_model
    if _spacy_nlp is not None:
        _spacy_nlp = None
        logger.info("Unloaded spaCy model")
    if _keybert_model is not None:
        del _keybert_model
        _keybert_model = None
        logger.info("Unloaded KeyBERT model")
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Cleared CUDA cache after keyword model unload")
    except ImportError:
        pass
