#!/bin/bash
# ================================================================
# train_per_constraint_ablation.sh — 逐约束消融训练（服务器版）
#
# 4 架构（ProfileNet/LSTM/CNN-1D/Transformer）× 4 单约束关闭配置
# （mono_off / bdy_off / smooth_off / log_off）共 16 次 Te 训练。
# 每次仅关闭一个约束，其余权重保持默认（mono 0.05, bdy 0.05,
# smooth 0.01, log 0.1）。
#
# 用法（服务器端）:
#   cd ~/fit
#   bash paper/train_per_constraint_ablation.sh
#
# 输出:
#   profile_nn_models_ablation_pc/{config}/{Arch}_Te.pt
#   profile_nn_models_ablation_pc/training_log.txt
# ================================================================

DATA_DIR="./profile_nn_data"
OUTPUT_BASE="./profile_nn_models_ablation_pc"
EPOCHS=500

# HPC SDK 冲突规避（torch 需要）
PYTORCH_RUN() {
    env LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v hpc_sdk | tr '\n' ':') python3 "$@"
}

CONFIGS=(mono_off bdy_off smooth_off log_off)
ARCHS=("profile_nn.train:ProfileNet"
       "profile_nn.lstm_baseline:LSTM"
       "profile_nn.cnn_baseline:CNN-1D"
       "profile_nn.transformer_baseline:Transformer")

mkdir -p "$OUTPUT_BASE"
echo "===== session $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$OUTPUT_BASE/training_log.txt"

for cfg in "${CONFIGS[@]}"; do
    OUT="$OUTPUT_BASE/$cfg"
    mkdir -p "$OUT"
    case "$cfg" in
        mono_off)   W_FLAGS="--w-mono 0" ;;
        bdy_off)    W_FLAGS="--w-bdy 0" ;;
        smooth_off) W_FLAGS="--w-smooth 0" ;;
        log_off)    W_FLAGS="--w-log 0" ;;
    esac
    for entry in "${ARCHS[@]}"; do
        mod="${entry%%:*}"; name="${entry##*:}"
        echo ""
        echo "============================================"
        echo "--- $name Te ($cfg)  $(date '+%H:%M:%S') ---"
        echo "============================================"
        PYTORCH_RUN -m "$mod" \
            --data "$DATA_DIR" --datatype Te \
            --epochs "$EPOCHS" --output "$OUT" \
            $W_FLAGS
    done
done

echo ""
echo "============================================"
echo "Per-constraint ablation training complete"
echo "Models in $OUTPUT_BASE/{mono_off,bdy_off,smooth_off,log_off}/"
echo "============================================"
