import networkx as nx
from pharmashield.models import VerificationResult, Verdict

class SymbolicConstraintEngine:
    def __init__(self, kg_graph: nx.DiGraph, contraindications_db: dict):
        self.graph = kg_graph
        self.contra_db = contraindications_db

    def check(self, normalized_claims: List[dict]) -> List[VerificationResult]:
        results = []
        for item in normalized_claims:
            claim = item["claim"]
            entities = item["entities"]
            
            if claim.type == "Contraindication":
                drug_cui = entities[0].cui
                disease_cui = entities[1].cui
                
                verdict, subsumed = self._evaluate_contraindication(drug_cui, disease_cui)
                results.append(VerificationResult(
                    claim_id=claim.id, verdict=verdict, confidence=1.0, 
                    rule_violated=f"Contra({drug_cui}, {disease_cui})", 
                    subsumption_used=subsumed
                ))
            else:
                results.append(VerificationResult(claim_id=claim.id, verdict=Verdict.UNCERTAIN, confidence=0.0))
        return results

    def _evaluate_contraindication(self, drug: str, disease: str) -> tuple:
        # Direct match
        if (drug, disease) in self.contra_db:
            return Verdict.INVALID, False
        
        # Ontological Subsumption: P ⊑ D => Contra(P, M)
        if disease in self.graph:
            for ancestor in nx.ancestors(self.graph, disease):
                if (drug, ancestor) in self.contra_db:
                    return Verdict.INVALID, True
                    
        return Verdict.UNCERTAIN, False
