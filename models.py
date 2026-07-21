from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Verdict(str, Enum):
    VALID = "Valid"
    INVALID = "Invalid"
    UNCERTAIN = "Uncertain"

class Severity(str, Enum):
    CRITICAL = "Critical"
    MAJOR = "Major"
    MODERATE = "Moderate"
    MINOR = "Minor"

class AtomicClaim(BaseModel):
    id: str
    type: str  # e.g., "DDI", "Contraindication"
    entities: List[str]
    raw_text: str

class NormalizedEntity(BaseModel):
    original: str
    cui: str
    ontology: str  # ATC, MONDO, UMLS

class VerificationResult(BaseModel):
    claim_id: str
    verdict: Verdict
    confidence: float = 0.0
    rule_violated: Optional[str] = None
    subsumption_used: bool = False
