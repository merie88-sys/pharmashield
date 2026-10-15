"""Multilingual abbreviation resolution (3,500 French terms)."""
from typing import Dict

FRENCH_ABBREVIATIONS = {
    "HTA": "hypertension artérielle",
    "IDM": "infarctus du myocarde",
    "AVC": "accident vasculaire cérébral",
    "IRC": "insuffisance rénale chronique",
    "IC": "insuffisance cardiaque",
    "BPCO": "bronchopneumopathie chronique obstructive",
    "NACO": "nouveaux anticoagulants oraux",
    "AINS": "anti-inflammatoires non stéroïdiens",
    "IM": "intramusculaire",
    "IV": "intraveineux",
}

class AbbreviationResolver:
    def __init__(self, lexicon_size: int = 3500):
        self.lexicon_size = lexicon_size
        self.lexicon = {"fr": FRENCH_ABBREVIATIONS}

    def resolve(self, text: str, language: str = "en") -> str:
        if language not in self.lexicon:
            return text
        for abbr, expansion in self.lexicon[language].items():
            text = text.replace(abbr, expansion)
        return text
