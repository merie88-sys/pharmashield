"""Ontological subsumption engine.

Implements the hierarchical inference at the core of PharmaShield:
if Contra(D, M) holds for a parent disease class D, then Contra(P, M)
is logically entailed for every subclass P such that P ⊑ D.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import networkx as nx


@dataclass
class SubsumptionResult:
    """Result of a subsumption lookup."""
    found: bool
    matched_ancestor: Optional[str]
    depth: int
    confidence: float
    audit_note: str


class OntologicalSubsumption:
    """Traverses MONDO / HPO hierarchies to infer implicit contraindications."""

    def __init__(self, ontology_graph: nx.DiGraph, max_depth: int = 3):
        self.graph = ontology_graph
        self.max_depth = max_depth

    def find_contraindication(
        self,
        condition_id: str,
        medication_id: str,
        contraindication_index: dict,
    ) -> SubsumptionResult:
        """Check for a contraindication by climbing the disease hierarchy.

        Args:
            condition_id: MONDO identifier of the patient condition.
            medication_id: ATC / DrugBank identifier of the medication.
            contraindication_index: mapping (ancestor_id, med_id) -> True.

        Returns:
            A SubsumptionResult describing whether an ancestor matched.
        """
        # Depth 0: direct match
        if (condition_id, medication_id) in contraindication_index:
            return SubsumptionResult(
                found=True,
                matched_ancestor=condition_id,
                depth=0,
                confidence=1.0,
                audit_note="Direct contraindication match.",
            )

        # Depth 1..max_depth: ancestor traversal
        for depth in range(1, self.max_depth + 1):
            for ancestor in self._ancestors_at_depth(condition_id, depth):
                if (ancestor, medication_id) in contraindication_index:
                    confidence = self._depth_confidence(depth)
                    return SubsumptionResult(
                        found=True,
                        matched_ancestor=ancestor,
                        depth=depth,
                        confidence=confidence,
                        audit_note=(
                            f"Inferred via subsumption: {condition_id} "
                            f"⊑ {ancestor} at depth {depth}."
                        ),
                    )

        return SubsumptionResult(
            found=False,
            matched_ancestor=None,
            depth=-1,
            confidence=0.0,
            audit_note="No contraindication found within traversal depth.",
        )

    def _ancestors_at_depth(self, node: str, depth: int) -> list:
        """Return all ancestors of `node` exactly `depth` edges away."""
        if depth == 1:
            return list(self.graph.successors(node))
        ancestors = set()
        for parent in self.graph.successors(node):
            ancestors.update(self._ancestors_at_depth(parent, depth - 1))
        return list(ancestors)

    @staticmethod
    def _depth_confidence(depth: int) -> float:
        """Deeper inferences carry lower confidence."""
        return max(0.5, 1.0 - 0.15 * depth)
