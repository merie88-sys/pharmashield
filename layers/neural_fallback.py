import numpy as np
from sklearn.neural_network import MLPClassifier
from pharmashield.models import VerificationResult, Verdict

class NeuralFallback:
    def __init__(self):
        # Mock trained MLP on DrugBank DDI dataset
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
        self.model.fit(np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([0, 1])) # Dummy training

    def resolve(self, results: List[VerificationResult]) -> List[VerificationResult]:
        for res in results:
            if res.verdict == Verdict.UNCERTAIN:
                # Extract features (mocked)
                features = np.array([[0.85, 0.15]]) 
                prob = self.model.predict_proba(features)[0][1]
                
                if prob > 0.7: # theta_safe
                    res.verdict = Verdict.INVALID
                    res.confidence = prob
                    res.rule_violated = "Probabilistic Neural Fallback"
        return results

class AuditTrailGenerator:
    def generate(self, results: List[VerificationResult]) -> str:
        report = "--- PHARMASHIELD AUDIT TRAIL ---\n"
        for r in results:
            status = "FLAGGED" if r.verdict == Verdict.INVALID else "CLEARED"
            method = "via Subsumption" if r.subsumption_used else "Direct Match" if r.rule_violated else "Neural Fallback"
            report += f"[{status}] Claim {r.claim_id}: {r.verdict} ({method}) | Rule: {r.rule_violated}\n"
        return report
