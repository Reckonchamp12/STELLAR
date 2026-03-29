from setuptools import setup, find_packages

setup(
    name="stellar-forecasting",
    version="1.0.0",
    description="STELLAR: Spectral-Temporal Ensemble Learning with Latent Adaptive Representations",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["jupyter>=1.0.0", "matplotlib>=3.7.0", "seaborn>=0.12.0"],
    },
)
