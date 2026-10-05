"""Global configuration for the PharmaShield pipeline."""
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PharmaShieldConfig:
    """Central configuration object for all four layers."""

    # Layer 1: Neural extraction
    extraction_model: str = "gpt-4o"
    temperature: float = 0.7
    max_claims: int = 64

    # Layer 2: Ontology grounding
    ontologies: list = field(
        default_factory=lambda: ["ATC", "MONDO", "UMLS", "HPO"]
    )
    linking_threshold: float = 0.75

    # Layer 3: Symbolic verification
    knowledge_bases: list = field(
        default_factory=lambda: ["DDInter", "DrugCentral", "Beers", "TGA"]
    )
    subsumption_max_depth: int = 3
    severity_threshold: str = "major"

    # Layer 4: Neural fallback
    fallback_model: str = "mlp_ddi"
    theta_safe: float = 0.7

    # Audit
    audit_enabled: bool = True


DEFAULT_CONFIG = PharmaShieldConfig()
