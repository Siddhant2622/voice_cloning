"""Voice Clone Detection Prototype — package setup."""
from setuptools import setup, find_packages

setup(
    name="voiceguard",
    version="0.1.0",
    packages=find_packages(where=".", include=["src*"]),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.0",
        "scipy>=1.12",
        "scikit-learn>=1.4",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "websockets>=12",
        "pydantic>=2",
        "python-dotenv>=1",
        "PyYAML>=6",
        "click>=8.1",
        "rich>=13",
        "tqdm>=4.66",
        "aiofiles>=23",
    ],
)
