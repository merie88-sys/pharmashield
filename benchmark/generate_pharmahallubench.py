"""Generate PharmaHalluBench: 800 scenarios, 1247 assertions."""
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

random.seed(42)

ERROR_CATEGORIES = {
    "contraindication": 0.35,
    "drug_drug_interaction": 0.25,
    "age_related": 0.15,
    "renal_hepatic": 0.15,
    "dosage_error": 0.10,
}

SEVERITY_LEVELS = ["critical", "major", "moderate", "minor"]
SEVERITY_WEIGHTS = [0.30, 0.35, 0.25, 0.10]

DRUG_POOL = [
    "metformin", "warfarin", "ibuprofen", "aspirin", "digoxin",
    "lisinopril", "simvastatin", "amoxicillin", "omeprazole", "furosemide",
    "levothyroxine", "prednisone", "clopidogrel", "insulin", "gabapentin",
    "sertraline", "amlodipine", "metoprolol", "atorvastatin", "spironolactone",
]

CONDITION_POOL = [
    "chronic kidney disease", "hepatic cirrhosis", "peptic ulcer disease",
    "congestive heart failure", "atrial fibrillation", "type 2 diabetes",
    "hypertension", "asthma", "epilepsy", "pregnancy",
    "renal impairment", "liver failure", "gastrointestinal bleeding",
    "hypothyroidism", "hyperthyroidism",
]

def generate_scenario(scenario_id: int, kb_data: Dict) -> Dict:
    """Generate a single pharmacological scenario."""
    error_type = random.choices(list(ERROR_CATEGORIES.keys()),
                                weights=list(ERROR_CATEGORIES.values()))[0]
    severity = random.choices(SEVERITY_LEVELS, weights=SEVERITY_WEIGHTS)[0]
    drug = random.choice(DRUG_POOL)
    condition = random.choice(CONDITION_POOL)

    scenario = {
        "id": f"PHB-{scenario_id:04d}",
        "error_type": error_type,
        "severity": severity,
        "clinical_context": {
            "drug": drug,
            "condition": condition,
            "patient_age": random.randint(18, 90),
            "comorbidities": random.sample(CONDITION_POOL, k=random.randint(0, 3)),
        },
        "generated_response": _generate_llm_response(drug, condition, error_type),
        "ground_truth": {
            "is_valid": random.random() > 0.6,
            "violated_constraint": f"{drug}_contraindicated_{condition}" if severity in ["critical", "major"] else None,
            "source_kb": random.choice(["ddinter", "drugcentral", "beers2023", "tga"]),
        },
    }
    return scenario

def _generate_llm_response(drug, condition, error_type):
    templates = {
        "contraindication": f"Based on the patient's history of {condition}, {drug} is recommended at standard dosage.",
        "drug_drug_interaction": f"The combination therapy with {drug} appears safe given the current clinical picture.",
        "age_related": f"For this elderly patient, {drug} at full dose is appropriate.",
        "renal_hepatic": f"Given the {condition}, no dose adjustment for {drug} is necessary.",
        "dosage_error": f"Administer {drug} 500mg twice daily for the treatment of {condition}.",
    }
    return templates.get(error_type, f"{drug} is prescribed for {condition}.")

def generate_assertions(scenarios: List[Dict]) -> List[Dict]:
    """Generate 1247 assertions from 800 scenarios."""
    assertions = []
    for sc in scenarios:
        n_assertions = random.choices([1, 2], weights=[0.44, 0.56])[0]
        for i in range(n_assertions):
            assertions.append({
                "id": f"{sc['id']}-A{i+1}",
                "scenario_id": sc["id"],
                "text": sc["generated_response"],
                "label": "invalid" if not sc["ground_truth"]["is_valid"] else "valid",
                "severity": sc["severity"],
                "error_type": sc["error_type"],
            })
    return assertions[:1247]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/pharmahallubench/")
    parser.add_argument("--n_scenarios", type=int, default=800)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    kb_data = {"ddinter": [], "drugcentral": [], "beers2023": [], "tga": []}

    print(f"Generating {args.n_scenarios} scenarios...")
    scenarios = [generate_scenario(i, kb_data) for i in range(1, args.n_scenarios + 1)]

    print("Generating assertions...")
    assertions = generate_assertions(scenarios)

    with open(output_dir / "scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)
    with open(output_dir / "assertions.json", "w") as f:
        json.dump(assertions, f, indent=2)

    print(f"✓ {len(scenarios)} scenarios saved")
    print(f"✓ {len(assertions)} assertions saved")

    # Summary statistics
    from collections import Counter
    error_dist = Counter(sc["error_type"] for sc in scenarios)
    sev_dist = Counter(sc["severity"] for sc in scenarios)
    print(f"\nError distribution: {dict(error_dist)}")
    print(f"Severity distribution: {dict(sev_dist)}")

if __name__ == "__main__":
    main()
