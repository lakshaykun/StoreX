from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11.setup_helpers import ParallelCompile
import pybind11
import os

# Enable parallel compilation
ParallelCompile("NPY_NUM_BUILD_JOBS").install()

# Get the list of source files
source_files = [
    "python/bindings.cpp",
    "src/vector_store.cpp",
    "src/similarity.cpp", 
    "src/storage.cpp",
    "src/search_engine.cpp",
    "src/metadata_filter.cpp",
]

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "storex",
        source_files,
        include_dirs=[
            "include/",
            "src/",
        ],
        define_macros=[("VERSION_INFO", '"0.1.0"')],
        cxx_std=17,
        # Optimization flags
        extra_compile_args=[
            "-O3", 
            "-march=native",
            "-ffast-math"
        ] if os.name != 'nt' else ["/O2"],
    ),
]

# Read README for long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "StoreX: High-performance vector database for ML workflows"

setup(
    name="storex",
    version="0.1.0",
    author="StoreX Team",
    author_email="your.email@example.com",
    description="High-performance vector database for ML workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lakshaykun/StoreX",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["storex"],
    package_dir={"storex": "python"},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "test": ["pytest", "numpy"],
        "dev": ["pytest", "numpy", "black", "mypy", "build"],
        "ml": ["sentence-transformers", "transformers", "torch"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
    ],
    keywords="vector database, machine learning, embeddings, similarity search",
)
