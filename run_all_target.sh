#!/bin/bash
# Office-Home 数据集所有域适应任务训练脚本
# 域索引: 0=Art, 1=Clipart, 2=Product, 3=RealWorld
# 流程: 1. 先训练源域模型 2. 再做目标域适应

# 设置设备 (cuda / mps / cpu)
DEVICE="cuda"

# 域名称（用于日志显示）
DOMAINS=("Art" "Clipart" "Product" "RealWorld")

echo "=========================================="
echo "Office-Home Domain Adaptation Pipeline"
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
    echo "Training source model: ${DOMAINS[$s]} (--s $s)"
    echo "----------------------------------------"
    
    python train_source_65.py --s $s --t $s --device $DEVICE
    
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
        echo "Adapting: ${DOMAINS[$s]} -> ${DOMAINS[$t]} (--s $s --t $t)"
        echo "----------------------------------------"
        
        python train_target_65.py --s $s --t $t --device $DEVICE
        
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
echo "=========================================="
