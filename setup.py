from setuptools import setup, find_packages

setup(
    name="llm_quantization_study",
    version="0.1.0",
    author="Andy Villamayor",
    description="Estudio de consumo energético en inferencia de LLMs",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "codecarbon>=2.3.0",
        "llama-cpp-python>=0.2.0",
        "numpy>=1.21,<2.0",
        "pandas>=2.1.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "tqdm>=4.66.0",
        "jupyter>=1.0.0",
    ],
)
