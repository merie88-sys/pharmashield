"""Layer 1: Neural claim extraction with multilingual abbreviation resolution."""
import re
from typing import List, Dict, Tuple
import spacy
from pharmashield.utils.abbreviations import AbbreviationResolver

class ClaimExtractor:
    """Extracts atomic pharmacological claims from clinical text."""

    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load("en_core_sci_sm")
        self.abbrev_resolver = AbbreviationResolver(
            lexicon_size=config.abbreviation_lexicon_size
        )
        self.claim_patterns = self._build_patterns()

    def _build_patterns(self):
        """Regex patterns for pharmacological assertions."""
        return {
            "contraindication": re.compile(
                r"(?P<drug>[\w\s-]+)\s+(?:is\s+)?(?:strictly\s+)?contraindicated\s+"
                r"(?:in|for|with)\s+(?P<condition>[\w\s-]+)", re.I),
            "interaction": re.compile(
                r"(?P<drug1>[\w\s-]+)\s+(?:interacts?|combined)\s+with\s+"
                r"(?P<drug2>[\w\s-]+)", re.I),
            "dosage": re.compile(
                r"(?P<drug>[\w\s-]+)\s+(?:dose|dosage)\s+(?:of|:)?\s+"
                r"(?P<dose>\d+\.?\d*)\s*(?P<unit>mg|g|mcg|IU)", re.I),
            "adjustment": re.compile(
                r"(?:adjust|reduce|avoid)\s+(?P<drug>[\w\s-]+)\s+"
                r"(?:in|for)\s+(?P<condition>[\w\s-]+)", re.I),
        }

    def extract(self, text: str, language: str = "en") -> List[Dict]:
        """Extract atomic claims from text."""
        text = self.abbrev_resolver.resolve(text, language)
        doc = self.nlp(text)
        claims = []
        for claim_type, pattern in self.claim_patterns.items():
            for match in pattern.finditer(text):
                claim = {
                    "text": match.group(0),
                    "type": claim_type,
                    "entities": {k: v for k, v in match.groupdict().items() if v},
                    "span": (match.start(), match.end()),
                    "language": language,
                }
                claims.append(claim)
        return claims

    def __call__(self, text: str, language: str = "en") -> List[Dict]:
        return self.extract(text, language)
