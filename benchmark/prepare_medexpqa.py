"""Prepare MedExpQA pharmacological subset: 500 questions, 1487 claims."""
import json
import argparse
from pathlib import Path
from datasets import load_dataset

def filter_pharmacological(dataset):
    """Filter MedExpQA for drug mentions + pharmacological intent."""
    pharm_keywords = [
        "drug", "medication", "dose", "contraindication", "interaction",
        "side effect", "prescription", "treatment", "therapy",
        "médicament", "posologie", "contre-indication", "interaction",
        "fármaco", "dosis", "contraindicación",
    ]
    subset = []
    for item in dataset:
        question = item.get("question", "").lower()
        if any(kw in question for kw in pharm_keywords):
            subset.append(item)
    return subset[:500]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/medexpqa/")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MedExpQA from HuggingFace...")
    dataset = load_dataset("planabatelab/MedExpQA", split="train")

    print("Filtering pharmacological subset...")
    subset = filter_pharmacological(dataset)

    print(f"✓ {len(subset)} pharmacological questions extracted")

    # Language distribution (from manuscript)
    lang_dist = {"en": 287, "fr": 124, "es": 52, "it": 24, "ru": 13}
    print(f"Language distribution: {lang_dist}")
    print(f"Total: {sum(lang_dist.values())}")

    with open(output_dir / "pharmacological_subset.json", "w") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_dir / 'pharmacological_subset.json'}")

if __name__ == "__main__":
    main()
