from pharmashield.models import AtomicClaim

class NeuralClaimExtractor:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def extract(self, clinical_text: str) -> List[AtomicClaim]:
        # TODO: Replace mock with actual LLM API call (e.g., OpenAI, HuggingFace)
        # prompt = f"Extract atomic claims from: {clinical_text}"
        # response = self.llm.chat.completions.create(...)
        
        return [
            AtomicClaim(id="C1", type="Contraindication", entities=["Ibuprofen", "Asthma"], raw_text="Patient asthmatique sous ibuprofène"),
            AtomicClaim(id="C2", type="DDI", entities=["Warfarin", "Aspirin"], raw_text="Warfarine et aspirine prescrites ensemble")
        ]
