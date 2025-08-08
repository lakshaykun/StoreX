#!/bin/bash

# Build script for StoreX Python bindings

set -e  # Exit on error

echo "=== Building StoreX Python Bindings ==="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Please run this script from the StoreX root directory."
    exit 1
fi

# Check if PyBind11 is installed
if ! python -c "import pybind11" 2>/dev/null; then
    echo "Installing PyBind11..."
    pip install pybind11
fi

# Install other dependencies
echo "Installing build dependencies..."
pip install build wheel setuptools numpy

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
find . -name "*.so" -delete

# Build in development mode
echo "Building extension..."
python setup.py build_ext --inplace

# Test import
echo "Testing import..."
if python -c "import storex; print('✓ Import successful'); print('Version:', storex.__version__)" 2>/dev/null; then
    echo "✓ Build successful!"
else
    echo "✗ Build failed - import test failed"
    exit 1
fi

# Install in development mode
echo "Installing in development mode..."
pip install -e .

# Run basic test
echo "Running basic functionality test..."
python -c "
import storex
import numpy as np

# Test basic functionality
store = storex.VectorStore()
store.insert([1.0, 2.0, 3.0], {'test': 'data'})
print(f'✓ Basic test passed - store size: {len(store)}')

# Test similarity
sim = storex.cosine_similarity([1.0, 0.0], [0.0, 1.0])
print(f'✓ Similarity test passed - result: {sim}')
"

echo ""
echo "=== Build Complete ==="
echo "You can now use 'import storex' in Python"
echo ""
echo "To run examples:"
echo "  python examples/basic_usage.py"
echo "  python examples/ml_integration.py"
echo ""
echo "To run tests:"
echo "  pip install pytest"  
echo "  python -m pytest python/tests/ -v"
echo ""
echo "To build wheel for distribution:"
echo "  python -m build"
