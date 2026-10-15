"""Layer 2: Ontology-grounded entity linking (ATC, MONDO, UMLS, HPO)."""
from typing import List, Dict, Optional, Tuple
import numpy as np
from pharmashield.utils.ontologies import OntologyLoader

class EntityLinker:
    """Links extracted entities to standardized biomedical ontologies."""

    ONTOLOGY_MAP = {
        "drug": ["atc", "umls"],
        "disease": ["mondo", "umls"],
        "symptom": ["hpo", "umls"],
        "condition": ["mondo", "hpo"],
    }

    def __init__(self, config):
        self.config = config
        self.ontologies = OntologyLoader(config.ontology)
        self.similarity_threshold = config.ontology.similarity_threshold

    def link(self, claim: Dict) -> Dict:
        """Link all entities in a claim to ontology concepts."""
        linked = claim.copy()
        linked["grounded_entities"] = {}
        for entity_name, entity_value in claim.get("entities", {}).items():
            entity_type = self._infer_type(entity_name)
            candidates = self._get_candidates(entity_value, entity_type)
            best = self._select_best(candidate_value=entity_value, candidates=candidates)
            linked["grounded_entities"][entity_name] = {
                "raw": entity_value,
                "concept_id": best[0] if best else None,
                "label": best[1] if best else None,
                "source": best[2] if best else None,
                "score": best[3] if best else None,
            }
        return linked

    def _infer_type(self, entity_name: str) -> str:
        for key in self.ONTOLOGY_MAP:
            if key in entity_name.lower():
                return key
        return "condition"

    def _get_candidates(self, value: str, entity_type: str) -> List[Tuple]:
        sources = self.ONTOLOGY_MAP.get(entity_type, ["umls"])
        candidates = []
        for src in sources:
            results = self.ontologies.search(value, source=src,
                                             limit=5)
            for r in results:
                candidates.append((r["id"], r["label"], src, r["score"]))
        return candidates

    def _select_best(self, candidate_value: str, candidates: List[Tuple]) -> Optional[Tuple]:
        if not candidates:
            return None
        filtered = [c for c in candidates if c[3] >= self.similarity_threshold]
        if not filtered:
            return candidates[0]
        return max(filtered, key=lambda x: x[3])

    def __call__(self, claim: Dict) -> Dict:
        return self.link(claim)
