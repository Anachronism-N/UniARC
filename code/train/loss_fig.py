import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# 训练完成后加载 TensorBoard 日志数据
log_dir = '/commondocument/group2/ASRCompare/code/train/log/hubert_ASR/version_2'  # 你的 TensorBoard 日志路径
# 根据日志可视化train loss 曲线
event_files = glob.glob(os.path.join(log_dir, 'events.*'))

from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator(event_files[-1])
ea.Reload()

# 读取 train_loss
train_loss_events = ea.Scalars('train_loss_epoch')
steps = [e.step for e in train_loss_events]
values = [e.value for e in train_loss_events]

# 读取 train_loss
val_loss_events = ea.Scalars('val_loss_epoch')
# steps = [e.step for e in train_loss_events]
val_values = [e.value for e in val_loss_events]

plt.figure(figsize=(8,5))
# 分别绘制训练损失和验证损失曲线
plt.plot(steps, values, label='Train Loss', marker='o')
plt.plot(steps, val_values, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curves')
plt.grid(True)
plt.legend()
plt.tight_layout()
# 确保figures目录存在
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/train_val_loss_curves.png', dpi=300)
plt.show()
