from setuptools import setup, find_packages

setup(
    name="ripple",
    version="0.0.1",
    description="ripple: modular physics-ML framework for wave PDEs",
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "pyyaml", "numpy", "matplotlib"],
    # Automatic discovery of packages
    packages=find_packages(),
    entry_points={"console_scripts": ["ripple=ripple.cli:main"]},
)
