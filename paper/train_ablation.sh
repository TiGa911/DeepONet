#!/bin/bash
# ================================================================
# train_ablation.sh — 物理约束消融实验训练
#
# 训练 4 个架构的 Te 模型，disable 所有物理约束（纯 MSE）。
# 每个模型约 100-300 epochs，CPU 训练，预计共 2-4 小时。
#
# 用法（服务器端）:
#   cd ~/fit
#   bash paper/train_ablation.sh
# ================================================================

DATA_DIR="./profile_nn_data"
OUTPUT_DIR="./profile_nn_models_ablation"
EPOCHS=500

# HPC SDK 冲突规避
PYTORCH_RUN() {
    env LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v hpc_sdk | tr '\n' ':') python3 "$@"
}

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "Ablation Training: Pure MSE (no physics constraints)"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# ProfileNet
echo ""
echo "--- ProfileNet Te (pure MSE) ---"
PYTORCH_RUN -m profile_nn.train \
    --data "$DATA_DIR" --datatype Te \
    --epochs "$EPOCHS" --output "$OUTPUT_DIR" \
    --w-mono 0 --w-bdy 0 --w-smooth 0 --w-log 0

# LSTM
echo ""
echo "--- LSTM Te (pure MSE) ---"
PYTORCH_RUN -m profile_nn.lstm_baseline \
    --data "$DATA_DIR" --datatype Te --epochs "$EPOCHS" \
    --output "$OUTPUT_DIR" \
    --w-mono 0 --w-bdy 0 --w-smooth 0

# CNN-1D
echo ""
echo "--- CNN-1D Te (pure MSE) ---"
PYTORCH_RUN -m profile_nn.cnn_baseline \
    --data "$DATA_DIR" --datatype Te --epochs "$EPOCHS" \
    --output "$OUTPUT_DIR" \
    --w-mono 0 --w-bdy 0 --w-smooth 0

# Transformer
echo ""
echo "--- Transformer Te (pure MSE) ---"
PYTORCH_RUN -m profile_nn.transformer_baseline \
    --data "$DATA_DIR" --datatype Te --epochs "$EPOCHS" \
    --output "$OUTPUT_DIR" \
    --w-mono 0 --w-bdy 0 --w-smooth 0

echo ""
echo "============================================"
echo "Ablation training complete"
echo "Models saved to respective output dirs"
echo "============================================"
