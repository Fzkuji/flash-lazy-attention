from setuptools import setup

setup(
    name="flash-lazy-attention",
    version="0.1.0",
    author="Fzkuji",
    author_email="fzkuji@gmail.com",
    description="Flash Lazy Attention: Triton implementation of SWAT/Lazy Attention",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Fzkuji/flash-lazy-attention",
    py_modules=["lazy_attention_triton"],
    install_requires=[
        "numpy",
        "torch",
        "triton",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)