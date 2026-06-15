import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


keyword_patterns = [
    ("has_artrose", r"\b(?:gon)?artrose\b"),
    ("has_fractuur", r"\bfractu\w*|\bbotbreuk\b"),
    ("has_trauma", r"\btrauma\b|\bgevallen\b|\bval\b|\bdistorsie\b|\bverdraai"),
    ("has_no_trauma", r"\bgeen trauma\b|\bniet traumatisch\b"),
    ("has_pain", r"\bpijn\b|\bklacht\w*\b|\blast\b"),
    ("has_swelling", r"\bzwelling\b|\bhydrops\b|\bdik\b"),
    ("has_meniscus", r"\bmeniscus\w*\b"),
    ("has_ligament", r"\bbandletsel\b|\bkruisband\b|\bcollaterale band\b"),
    ("has_locking", r"\bslotklacht\w*\b|\bslot\b|\bblokk"),
    ("has_instability", r"\binstabiel\b|\binstabiliteit\b|\bdoorzak"),
    ("has_prosthesis", r"\bprothese\b|\btka\b|\barthroplastiek\b"),
    ("has_tumor_infection", r"\btumou?r\b|\bmalign|\bosteomyelitis\b|\binfect"),
]


class ReferralTextFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, include_length=True, include_keywords=True):
        self.include_length = include_length
        self.include_keywords = include_keywords

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        rows = []
        for value in x:
            text = "" if value is None else str(value)
            text_lower = text.lower()
            words = re.findall(r"\w+", text_lower)

            features = []
            if self.include_length:
                features.extend(
                    [
                        np.log1p(len(text)),
                        np.log1p(len(words)),
                        text.count("?"),
                        text.count("."),
                    ]
                )
            if self.include_keywords:
                features.extend(
                    1.0 if re.search(pattern, text_lower) else 0.0
                    for _, pattern in keyword_patterns
                )
            rows.append(features)

        if not rows:
            return sparse.csr_matrix((0, self.feature_count()))
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))

    def feature_count(self):
        total = 0
        if self.include_length:
            total += 4
        if self.include_keywords:
            total += len(keyword_patterns)
        return total

    def get_feature_names_out(self, input_features=None):
        names = []
        if self.include_length:
            names.extend(["text_length", "word_count", "question_marks", "sentence_count"])
        if self.include_keywords:
            names.extend(name for name, _ in keyword_patterns)
        return np.asarray(names, dtype=object)


def build_feature_extractor(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    use_manual_features=True,
):
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
    )
    if not use_manual_features:
        return tfidf
    return FeatureUnion(
        [
            ("tfidf", tfidf),
            ("manual", ReferralTextFeatures()),
        ]
    )
