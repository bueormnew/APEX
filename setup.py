from setuptools import setup, find_packages

setup(
    name="apex-core",
    version="0.1.0",
    description="APEX: High-Performance Hybrid Autoregressive Sequence Modeling Library (Hop-Mix + LRCM + Mamba-3 + Native ECHO)",
    author="APEX Team",
    packages=find_packages(),
    py_modules=[
        "echo", "hopmix", "lrcm", "mamba3", "hybrid_model", "apex_triton", "apex_cli"
    ],
    install_requires=[
        "torch>=2.0.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "cuda": [
            "triton>=2.1; platform_system == 'Linux'",
        ],
    },
    entry_points={
        "console_scripts": [
            "apex=apex_cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
