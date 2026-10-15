"""Global configuration for PharmaShield."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class OntologyConfig:
    atc_path: str = "data/ontologies/atc.owl"
    mondo_path: str = "data/ontologies/mondo.owl"
    umls_path: str = "data/ontologies/umls"
    hpo_path: str = "data/ontologies/hp.owl"
    embedding_dim: int = 384
    similarity_threshold: float = 0.85

@dataclass
class SymbolicConfig:
    theta_safe: float = 0.7
    max_subsumption_depth: int = 10
    enable_subsumption: bool = True
    knowledge_sources: List[str] = field(default_factory=lambda: [
        "ddinter", "drugcentral", "mondo", "beers2023", "tga"
    ])

@dataclass
class NeuralConfig:
    fallback_model: str = "distilbert-base-multilingual-cased"
    mlp_hidden_dim: int = 256
    mlp_dropout: float = 0.2
    confidence_threshold: float = 0.7

@dataclass
class PipelineConfig:
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    languages: List[str] = field(default_factory=lambda: ["en", "fr", "es", "it", "ru"])
    abbreviation_lexicon_size: int = 3500
