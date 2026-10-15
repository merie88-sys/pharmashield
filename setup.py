from setuptools import setup, find_packages

setup(
    name="pharmashield",
    version="1.0.0",
    description="Neuro-symbolic pharmacological verification for clinical LLMs",
    author="Meriem Kermani",
    author_email="meriem.kermani@univ-constantine2.dz",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0", "pandas>=2.0.0", "scipy>=1.11.0",
        "scikit-learn>=1.3.0", "matplotlib>=3.7.0", "transformers>=4.35.0",
        "torch>=2.1.0", "owlready2>=0.45", "networkx>=3.1",
        "spacy>=3.7.0", "tqdm>=4.66.0", "pyyaml>=6.0",
    ],
    extras_require={
        "llm": ["openai>=1.0.0", "huggingface-hub>=0.19.0"],
        "dev": ["pytest>=7.4.0", "black", "flake8"],
    },
)
