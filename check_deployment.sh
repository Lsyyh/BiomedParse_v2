#!/bin/bash
# BiomedParse v2 Deployment Check Script

set -e

echo "=== BiomedParse v2 Deployment Check ==="
echo ""

ERRORS=0
WARNINGS=0

# Function to check Python package
check_package() {
    if conda run -n biomedparse_v2 python3 -c "import $1" 2>/dev/null; then
        VERSION=$(conda run -n biomedparse_v2 python3 -c "import $1; print($1.__version__)" 2>/dev/null || echo "unknown")
        echo "✓ $1 is installed (version: $VERSION)"
        return 0
    else
        echo "✗ $1 is not installed"
        return 1
    fi
}

# Function to check file
check_file() {
    if [ -f "$1" ]; then
        echo "✓ File exists: $1"
        return 0
    else
        echo "✗ File not found: $1"
        return 1
    fi
}

echo "1. Checking conda environment..."
if conda env list | grep -q "biomedparse_v2"; then
    echo "✓ biomedparse_v2 environment exists"
else
    echo "✗ biomedparse_v2 environment not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

echo "2. Checking Python packages..."
check_package "torch" || ERRORS=$((ERRORS + 1))
check_package "torchvision" || ERRORS=$((ERRORS + 1))
check_package "fastapi" || ERRORS=$((ERRORS + 1))
check_package "uvicorn" || ERRORS=$((ERRORS + 1))
check_package "huggingface_hub" || ERRORS=$((ERRORS + 1))
check_package "PIL" || ERRORS=$((ERRORS + 1))
check_package "numpy" || ERRORS=$((ERRORS + 1))
check_package "hydra" || ERRORS=$((ERRORS + 1))
echo ""

echo "3. Checking CUDA availability..."
if conda run -n biomedparse_v2 python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✓ CUDA is available"
    GPU_NAME=$(conda run -n biomedparse_v2 python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "  GPU: $GPU_NAME"
else
    echo "⚠ CUDA not available (CPU mode will be used)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

echo "4. Checking HuggingFace login..."
if conda run -n biomedparse_v2 huggingface-cli whoami 2>/dev/null | grep -q "Not logged in"; then
    echo "⚠ Not logged in to HuggingFace"
    echo "  Please login: huggingface-cli login"
    WARNINGS=$((WARNINGS + 1))
else
    echo "✓ HuggingFace login OK"
fi
echo ""

echo "5. Checking service files..."
check_file "/home/data/zgb/Code/BiomedParse-2/biomedparse_segmentation_service.py" || ERRORS=$((ERRORS + 1))
check_file "/home/data/zgb/Code/BiomedParse-2/start_segmentation_service.sh" || WARNINGS=$((WARNINGS + 1))
check_file "/home/data/zgb/Code/BiomedParse-2/configs/model/biomedparse.yaml" || ERRORS=$((ERRORS + 1))
check_file "/home/data/zgb/Code/BiomedParse-2/src/model/biomedparse.py" || ERRORS=$((ERRORS + 1))
echo ""

echo "=== Summary ==="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo "✓ Deployment check passed!"
    echo ""
    echo "To start the service:"
    echo "  cd /home/data/zgb/Code/BiomedParse-2"
    echo "  ./start_segmentation_service.sh"
    echo ""
    echo "Or manually:"
    echo "  conda activate biomedparse_v2"
    echo "  python3 biomedparse_segmentation_service.py"
else
    echo "✗ Deployment check failed!"
    echo "Please fix the errors above before starting the service."
    exit 1
fi
