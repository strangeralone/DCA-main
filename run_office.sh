#!/bin/bash
# Office-31 数据集所有域适应任务训练脚本
# 使用新的 main.py 统一入口
#
# 用法:
#   ./run_office.sh [方法] [设备]
#   ./run_office.sh dca cuda
#   ./run_office.sh dca_clip mps
#
# 域索引: 0=amazon, 1=dslr, 2=webcam

# 解析参数
METHOD="${1:-dca}"
DEVICE="${2:-cuda}"

# 域名称（用于日志显示）
DOMAINS=("amazon" "dslr" "webcam")

echo "=========================================="
echo "DCA Training Pipeline - Office-31"
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

for s in 0 1 2; do
    echo ""
    echo "----------------------------------------"
    echo "Training source model: ${DOMAINS[$s]} (--source $s)"
    echo "----------------------------------------"
    
    python main.py --method $METHOD --dataset office --source $s --mode source --device $DEVICE
    
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
for s in 0 1 2; do
    # 遍历所有目标域
    for t in 0 1 2; do
        # 跳过 s == t 的情况
        if [ $s -eq $t ]; then
            continue
        fi
        
        echo ""
        echo "----------------------------------------"
        echo "Adapting: ${DOMAINS[$s]} -> ${DOMAINS[$t]} (--source $s --target $t)"
        echo "----------------------------------------"
        
        python main.py --method $METHOD --dataset office --source $s --target $t --mode target --device $DEVICE
        
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
echo "Output directory: ckps/office/$METHOD/"
echo "=========================================="
