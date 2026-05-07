#!/bin/bash
# BiomedParse v2 Segmentation Service Startup Script

set -e

echo "=== BiomedParse v2 Segmentation Service ==="
echo ""

# Check HuggingFace login
echo "Checking HuggingFace login..."
if conda run -n biomedparse_v2 huggingface-cli whoami 2>/dev/null | grep -q "Not logged in"; then
    echo "WARNING: Not logged in to HuggingFace"
    echo "Please login first:"
    echo "  conda activate biomedparse_v2"
    echo "  huggingface-cli login"
    echo ""
fi

echo "Starting BiomedParse v2 segmentation service..."
echo "Service will be available at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""

# Start service
conda run -n biomedparse_v2 python3 biomedparse_segmentation_service.py
