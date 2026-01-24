#!/bin/bash
# Office-Home 数据集所有域适应任务训练脚本（重构版）
# 使用新的 main.py 统一入口
#
# 用法:
#   ./run_all.sh [方法] [设备]
#   ./run_all.sh dca cuda
#   ./run_all.sh dca_clip mps
#
# 域索引: 0=Art, 1=Clipart, 2=Product, 3=RealWorld

# 解析参数
METHOD="${1:-dca_coop}"
DEVICE="${2:-cuda}"

# 域名称（用于日志显示）
DOMAINS=("Art" "Clipart" "Product" "RealWorld")

echo "=========================================="
echo "DCA Training Pipeline (Refactored)"
echo "=========================================="
echo "Method: $METHOD"
echo "Device: $DEVICE"
echo "=========================================="

# ============================================
# 第一阶段：训练所有源域模型
# ============================================
echo ""
echo "=========================================="
echo "Phase 1: Source Domain Training"
echo "=========================================="

for s in 0 1 2 3; do
    echo ""
    echo "----------------------------------------"
    echo "Training source model: ${DOMAINS[$s]} (--source $s)"
    echo "----------------------------------------"
    
    python main.py --method $METHOD --dataset officehome --source $s --mode source --device $DEVICE
    
    if [ $? -ne 0 ]; then
        echo "Error: Source training for ${DOMAINS[$s]} failed!"
        exit 1
    fi
    
    echo "Completed: Source model for ${DOMAINS[$s]}"
done

# ============================================
# 第二阶段：目标域适应
# ============================================
echo ""
echo "=========================================="
echo "Phase 2: Target Domain Adaptation"
echo "=========================================="

# 遍历所有源域
for s in 0 1 2 3; do
    # 遍历所有目标域
    for t in 0 1 2 3; do
        # 跳过 s == t 的情况
        if [ $s -eq $t ]; then
            continue
        fi
        
        echo ""
        echo "----------------------------------------"
        echo "Adapting: ${DOMAINS[$s]} -> ${DOMAINS[$t]} (--source $s --target $t)"
        echo "----------------------------------------"
        
        python main.py --method $METHOD --dataset officehome --source $s --target $t --mode target --device $DEVICE
        
        if [ $? -ne 0 ]; then
            echo "Error: ${DOMAINS[$s]} -> ${DOMAINS[$t]} failed!"
            exit 1
        fi
        
        echo "Completed: ${DOMAINS[$s]} -> ${DOMAINS[$t]}"
    done
done

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "Output directory: ckps/officehome/$METHOD/"
echo "=========================================="
