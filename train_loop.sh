#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Number of generations to train
GENERATIONS=25
ENV_NAME="hex-env"

echo "========================================"
echo "🚀 Starting AlphaZero Training Pipeline"
echo "========================================"

for ((i=1; i<=GENERATIONS; i++))
do
    echo ""
    echo "▶️ GENERATION $i / $GENERATIONS"
    
    # 1. Generate new self-play data
    echo "Step 1: Generating self-play data..."
    mamba run -n $ENV_NAME python3 -m src.agent.self_play
    
    # 2. Train the PyTorch model on the new data
    echo "Step 2: Training neural network..."
    mamba run -n $ENV_NAME python3 -m src.agent.train
    
    # 3. Export to ONNX so the NEXT generation of self-play uses the smarter weights
    echo "Step 3: Exporting new ONNX model..."
    mamba run -n $ENV_NAME python3 -m src.agent.export_onnx
    
    # 4. Clean up old data so we don't train on outdated garbage
    echo "Step 4: Wiping old experience replay buffer..."
    rm -rf training/data/*.pkl
    
    # 5. Update the metadata JSON for the Streamlit UI
    echo "Step 5: Updating UI metadata ledger..."
    TOTAL_GAMES=$((i * 100))
    
    # Logic gate to update the UI text based on mathematical thresholds
    if [ $i -ge 26 ]; then
        STATUS="Phase 2 (Emergence) - Basic clustering and center control"
    else
        STATUS="Phase 1 (The Bootstrap) - High-entropy noise"
    fi
    
    # Write directly to the JSON file
    cat <<EOF > models/production/metadata.json
{
    "generation": $i,
    "total_games": $TOTAL_GAMES,
    "status": "$STATUS",
    "architecture": "AlphaZero ResNet Dual-Head"
}
EOF
    
    echo "✅ Generation $i complete."
done

echo ""
echo "🎉 Training pipeline finished successfully!"