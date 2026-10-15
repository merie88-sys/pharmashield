"""Layer 3: Deterministic constraint checking + ontological subsumption."""
from typing import Dict, List, Optional, Tuple
from enum import Enum

class Verdict(Enum):
    VALID = "Valid"
    INVALID = "Invalid"
    UNCERTAIN = "Uncertain"

class SymbolicVerifier:
    """Deterministic constraint checker with ontological subsumption."""

    def __init__(self, config, knowledge_base):
        self.config = config.symbolic
        self.kb = knowledge_base
        self.audit_trail = []

    def verify(self, claim: Dict) -> Dict:
        """Verify a grounded claim against pharmacological constraints."""
        result = {
            "verdict": Verdict.UNCERTAIN.value,
            "rule_id": None,
            "source": None,
            "mechanism": None,
            "audit": [],
        }
        drug_id = claim["grounded_entities"].get("drug", {}).get("concept_id")
        condition_id = claim["grounded_entities"].get("condition", {}).get("concept_id")
        if not drug_id or not condition_id:
            return result

        # Direct constraint check
        direct = self._check_direct(drug_id, condition_id)
        if direct:
            result.update(direct)
            result["mechanism"] = "direct_match"
            return result

        # Ontological subsumption
        if self.config.enable_subsumption:
            subsumed = self._check_subsumption(drug_id, condition_id)
            if subsumed:
                result.update(subsumed)
                result["mechanism"] = "ontological_subsumption"
                return result

        return result

    def _check_direct(self, drug_id: str, condition_id: str) -> Optional[Dict]:
        """Check explicit contraindications in knowledge base."""
        for source in self.config.knowledge_sources:
            rule = self.kb.lookup(drug_id, condition_id, source=source)
            if rule:
                return {
                    "verdict": Verdict.INVALID.value,
                    "rule_id": rule["id"],
                    "source": source,
                    "audit": [rule],
                }
        return None

    def _check_subsumption(self, drug_id: str, condition_id: str) -> Optional[Dict]:
        """Traverse MONDO/HPO hierarchy for implicit contraindications."""
        ancestors = self.kb.get_ancestors(condition_id,
                                          max_depth=self.config.max_subsumption_depth)
        for ancestor_id, depth in ancestors:
            rule = self.kb.lookup(drug_id, ancestor_id, source="mondo")
            if rule:
                return {
                    "verdict": Verdict.INVALID.value,
                    "rule_id": rule["id"],
                    "source": "mondo_subsumption",
                    "subsumption_path": {
                        "patient_condition": condition_id,
                        "inferred_parent": ancestor_id,
                        "depth": depth,
                        "relation": "P_subsumes_D",
                    },
                    "audit": [rule, {"type": "subsumption", "depth": depth}],
                }
        return None

    def __call__(self, claim: Dict) -> Dict:
        return self.verify(claim)
