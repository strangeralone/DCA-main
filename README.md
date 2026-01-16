# source
python main.py --method dca --source 0 --mode source


# target

# 2. 目标域适应（从 Art 适应到 Clipart）
python main.py --method dca_clip --source 0 --target 1 --mode target
# 3. 或者一步完成源域训练 + 目标适应
python main.py --method dca --source 0 --target 1 --mode all