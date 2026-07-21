
import networkx as nx
from pharmashield.layers.neural_extraction import NeuralClaimExtractor
from pharmashield.layers.entity_linking import EntityLinker
from pharmashield.layers.symbolic_engine import SymbolicConstraintEngine
from pharmashield.layers.neural_fallback import NeuralFallback
from pharmashield.layers.audit_trail import AuditTrailGenerator

def run_pharmashield(clinical_text: str):
    # 1. Initialize Mock Knowledge Graph (MONDO hierarchy)
    kg_graph = nx.DiGraph()
    kg_graph.add_edges_from([("Asthma", "Respiratory_Disease"), ("Respiratory_Disease", "Disease")])
    
    contra_db = {
        ("ATC_Ibuprofen", "Respiratory_Disease"): True # Ibuprofen contraindicated for Respiratory Diseases
    }
    
    kg_index = {"ibuprofen": "ATC_Ibuprofen", "asthma": "Asthma", "warfarin": "ATC_Warfarin", "aspirin": "ATC_Aspirin"}

    # 2. Instantiate Layers
    layer1 = NeuralClaimExtractor()
    layer2 = EntityLinker(kg_index)
    layer3 = SymbolicConstraintEngine(kg_graph, contra_db)
    layer4_fallback = NeuralFallback()
    layer4_audit = AuditTrailGenerator()

    # 3. Execute Pipeline
    claims = layer1.extract(clinical_text)
    normalized = layer2.normalize(claims)
    symbolic_results = layer3.check(normalized)
    final_results = layer4_fallback.resolve(symbolic_results)
    
    # 4. Output
    print(layer4_audit.generate(final_results))

if __name__ == "__main__":
    run_pharmashield("Patient asthmatique sous ibuprofène et warfarine avec aspirine.")
