"""End-to-end PharmaShield verification pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pharmashield.config import PharmaShieldConfig, DEFAULT_CONFIG
from pharmashield.layer1_extraction.claim_extractor import ClaimExtractor
from pharmashield.layer2_grounding.entity_linker import EntityLinker
from pharmashield.layer3_symbolic.symbolic_engine import SymbolicEngine
from pharmashield.layer4_fallback.neural_fallback import NeuralFallback
from pharmashield.audit.audit_trail import AuditTrailGenerator


@dataclass
class VerifiedClaim:
    """A single verified atomic claim with its verdict and audit trail."""
    text: str
    claim_type: str
    verdict: str            # "Valid" | "Invalid" | "Uncertain"
    confidence: float
    audit_trail: str


@dataclass
class VerificationResult:
    """Aggregated output of the pipeline."""
    claims: List[VerifiedClaim]
    n_invalid: int
    n_valid: int
    n_uncertain: int


class PharmaShieldPipeline:
    """Orchestrates the four verification layers."""

    def __init__(self, config: PharmaShieldConfig = DEFAULT_CONFIG):
        self.config = config
        self.extractor = ClaimExtractor(config)
        self.linker = EntityLinker(config)
        self.symbolic_engine = SymbolicEngine(config)
        self.fallback = NeuralFallback(config)
        self.audit_generator = AuditTrailGenerator(config)

    @classmethod
    def from_pretrained(cls, config_path: str) -> "PharmaShieldPipeline":
        import yaml
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(PharmaShieldConfig(**data))

    def verify(self, clinical_text: str) -> VerificationResult:
        """Run the full Neuro -> Symbolic -> Neural verification pipeline."""
        # Layer 1: neural claim extraction
        raw_claims = self.extractor.extract(clinical_text)

        # Layer 2: ontology grounding
        grounded_claims = self.linker.link_all(raw_claims)

        verified: List[VerifiedClaim] = []
        for claim in grounded_claims:
            # Layer 3: deterministic symbolic verification
            symbolic = self.symbolic_engine.check(claim)

            if symbolic.verdict != "Uncertain":
                verdict, confidence = symbolic.verdict, symbolic.confidence
            else:
                # Layer 4: neural fallback only when symbolic is silent
                prob = self.fallback.predict(claim)
                if prob > self.config.theta_safe:
                    verdict, confidence = "Invalid", prob
                else:
                    verdict, confidence = "Valid", 1.0 - prob

            audit = self.audit_generator.generate(claim, verdict, symbolic)
            verified.append(
                VerifiedClaim(
                    text=claim.text,
                    claim_type=claim.type,
                    verdict=verdict,
                    confidence=confidence,
                    audit_trail=audit,
                )
            )

        return VerificationResult(
            claims=verified,
            n_invalid=sum(c.verdict == "Invalid" for c in verified),
            n_valid=sum(c.verdict == "Valid" for c in verified),
            n_uncertain=sum(c.verdict == "Uncertain" for c in verified),
        )
