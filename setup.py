from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPI_README = README.replace(
    "](./assets/",
    "](https://raw.githubusercontent.com/wooyeolbaek/attention-map-diffusers/main/assets/",
)

setup(
    name="attention_map_diffusers",
    version="1.0.0",
    author="Wooyeol Baek, Seungyeol Baek",
    author_email="wooyeol.baek@yonsei.ac.kr, mbaek01@korea.ac.kr",
    description=(
        "Extract and visualize relation-aware attention maps from "
        "Hugging Face Diffusers pipelines"
    ),
    long_description=PYPI_README,
    long_description_content_type="text/markdown",
    url="https://github.com/wooyeolBaek/attention-map-diffusers",
    project_urls={
        "Documentation": "https://github.com/wooyeolBaek/attention-map-diffusers#readme",
        "Source": "https://github.com/wooyeolBaek/attention-map-diffusers",
        "Issues": "https://github.com/wooyeolBaek/attention-map-diffusers/issues",
        "DOI": "https://doi.org/10.5281/zenodo.18304023",
    },
    license="MIT",
    license_files=["LICENSE"],
    packages=find_packages(exclude=("demo", "demo.*", "tests", "tests.*")),
    include_package_data=True,
    keywords=[
        "attention maps",
        "diffusers",
        "diffusion models",
        "interpretability",
        "text-to-image",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "diffusers>=0.39.0",
        "accelerate",
        "transformers>=4.57.0",
        "ftfy",
        "numpy",
        "Pillow",
        "torch>=2.0.0",
        "torchvision",
        "protobuf",
        "sentencepiece",
    ],
    extras_require={
        "dev": ["build", "pytest", "twine"],
    },
)
