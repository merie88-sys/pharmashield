"""Layer 4: Neural fallback (MLP) for symbolically uncertain cases."""
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from pharmashield.layer3_symbolic_verification import Verdict

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)

class NeuralFallback:
    """MLP fallback triggered only on symbolic uncertainty."""

    def __init__(self, config, embedding_model):
        self.config = config.neural
        self.embedding_model = embedding_model
        self.classifier = MLPClassifier(
            input_dim=config.ontology.embedding_dim,
            hidden_dim=self.config.mlp_hidden_dim,
            dropout=self.config.mlp_dropout,
        )
        self.classifier.eval()
        self.triggered = 0
        self.flagged = 0
        self.released = 0

    def should_trigger(self, symbolic_result: Dict) -> bool:
        return symbolic_result["verdict"] == Verdict.UNCERTAIN.value

    def predict(self, claim: Dict) -> Dict:
        """Neural prediction for uncertain claims."""
        self.triggered += 1
        text = claim.get("text", "")
        embedding = self.embedding_model.encode(text)
        tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.classifier(tensor)
            probs = torch.softmax(logits, dim=-1)
        risk_prob = probs[0][1].item()
        is_invalid = risk_prob > self.config.confidence_threshold
        if is_invalid:
            self.flagged += 1
        else:
            self.released += 1
        return {
            "verdict": Verdict.INVALID.value if is_invalid else Verdict.VALID.value,
            "p_risk": risk_prob,
            "mechanism": "neural_fallback",
            "triggered": True,
        }

    def get_stats(self) -> Dict:
        return {
            "triggered": self.triggered,
            "flagged": self.flagged,
            "released": self.released,
        }
