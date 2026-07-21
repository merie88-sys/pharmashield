# PharmaShield: Neuro-Symbolic Verification for Pharmacological Hallucination Mitigation

Official implementation of the **PharmaShield** framework, a four-layer neuro-symbolic architecture designed to transform LLM-generated clinical recommendations into formally verifiable, interpretable, and auditable decisions. 

This repository contains the core algorithmic pipeline, including the ontological subsumption mechanism and the hybrid Neuro $\rightarrow$ Symbolic $\rightarrow$ Neuro verification engine.

## 📖 Overview

PharmaShield addresses the critical issue of LLM hallucinations in clinical decision support by decoupling semantic extraction from formal constraint checking. The architecture operates through four integrated layers:

1. **Neural Claim Extraction:** Parses atomic clinical entities from unstructured text.
2. **Ontology-Grounding:** Normalizes entities to standardized biomedical identifiers (ATC, MONDO, UMLS).
3. **Symbolic Constraint Checking:** Validates recommendations against pharmacological knowledge graphs using deterministic rules and **ontological subsumption** ($P \sqsubseteq D \Rightarrow \text{Contra}(P, M)$).
4. **Neural Fallback & Audit:** Resolves uncertainty via a probabilistic MLP and generates an interpretable clinical audit trail.

## 📁 Project Structure

```text
pharmashield/
├── main.py                      # Main execution pipeline
├── requirements.txt             # Python dependencies
├── pharmashield/
│   ├── __init__.py
│   ├── models.py                # Pydantic data models
│   ├── config.py                # Configuration & thresholds
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── neural_extraction.py # Layer 1: LLM-based parsing
│   │   ├── entity_linking.py    # Layer 2: Ontology mapping
│   │   ├── symbolic_engine.py   # Layer 3: Deterministic verification & Subsumption
│   │   ├── neural_fallback.py   # Layer 4: Probabilistic resolution
│   │   └── audit_trail.py       # Layer 4: Report generation
│   ── knowledge_graph/
│       ├── __init__.py
│       └── kg_loader.py         # Knowledge graph & ontology utilities
