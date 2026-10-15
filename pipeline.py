"""Main PharmaShield pipeline orchestrating all four layers."""
from typing import List, Dict
import time
from pharmashield.layer1_claim_extraction import ClaimExtractor
from pharmashield.layer2_entity_linking import EntityLinker
from pharmashield.layer3_symbolic_verification import SymbolicVerifier, Verdict
from pharmashield.layer4_neural_fallback import NeuralFallback
from pharmashield.utils.metrics import compute_metrics

class PharmaShieldPipeline:
    """End-to-end neuro-symbolic verification pipeline."""

    def __init__(self, config, knowledge_base, embedding_model):
        self.config = config
        self.layer1 = ClaimExtractor(config)
        self.layer2 = EntityLinker(config)
        self.layer3 = SymbolicVerifier(config, knowledge_base)
        self.layer4 = NeuralFallback(config, embedding_model)
        self.latency_log = {"layer1": [], "layer2": [], "layer3": [], "layer4": []}

    def verify_text(self, text: str, language: str = "en") -> Dict:
        """Full pipeline: text -> verdict + audit trail."""
        t0 = time.perf_counter()

        # Layer 1
        claims = self.layer1(text, language)
        self.latency_log["layer1"].append(time.perf_counter() - t0)

        results = []
        for claim in claims:
            # Layer 2
            t1 = time.perf_counter()
            grounded = self.layer2(claim)
            self.latency_log["layer2"].append(time.perf_counter() - t1)

            # Layer 3
            t2 = time.perf_counter()
            verdict = self.layer3(grounded)
            self.latency_log["layer3"].append(time.perf_counter() - t2)

            # Layer 4 (conditional)
            if self.layer4.should_trigger(verdict):
                t3 = time.perf_counter()
                verdict = self.layer4.predict(grounded)
                self.latency_log["layer4"].append(time.perf_counter() - t3)

            results.append(verdict)

        final = self._aggregate(results)
        return final

    def _aggregate(self, results: List[Dict]) -> Dict:
        invalid_count = sum(1 for r in results if r["verdict"] == Verdict.INVALID.value)
        return {
            "final_verdict": Verdict.INVALID.value if invalid_count > 0 else Verdict.VALID.value,
            "n_claims": len(results),
            "n_invalid": invalid_count,
            "claims": results,
            "audit_trail": [r.get("audit", []) for r in results],
        }

    def evaluate(self, dataset: List[Dict]) -> Dict:
        """Evaluate pipeline on a labeled dataset."""
        predictions = []
        ground_truth = []
        for sample in dataset:
            pred = self.verify_text(sample["text"], sample.get("language", "en"))
            predictions.append(1 if pred["final_verdict"] == Verdict.INVALID.value else 0)
            ground_truth.append(1 if sample["label"] == "invalid" else 0)
        return compute_metrics(ground_truth, predictions)
