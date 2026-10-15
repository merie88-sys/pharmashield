"""Run main PharmaHalluBench experiments."""
import json
import argparse
from pathlib import Path
from pharmashield.pipeline import PharmaShieldPipeline
from pharmashield.config import PipelineConfig
from pharmashield.utils.metrics import compute_metrics, bootstrap_ci

def load_benchmark(path):
    with open(path / "assertions.json") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/pharmahallubench/")
    parser.add_argument("--output", type=str, default="results/main_results.json")
    args = parser.parse_args()

    config = PipelineConfig()
    dataset = load_benchmark(Path(args.data))
    print(f"Loaded {len(dataset)} assertions")

    pipeline = PharmaShieldPipeline(config, knowledge_base=None, embedding_model=None)
    results = pipeline.evaluate(dataset)

    # Add bootstrap confidence intervals
    f1_ci = bootstrap_ci(results["f1"], n_bootstrap=10000)
    results["f1_ci"] = f1_ci

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"F1: {results['f1']:.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
    print(f"CFNR: {results.get('cfnr', 'N/A')}")

if __name__ == "__main__":
    main()
