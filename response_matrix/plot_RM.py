import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
from cycler import cycler

# 设置字体大小和绘图样式
rcParams['font.size'] = 18
rcParams['axes.titlesize'] = 20
rcParams['axes.labelsize'] = 18
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 16
rcParams['figure.titlesize'] = 22
plt.style.use('seaborn-v0_8-whitegrid')

# 文件路径
rm_file = os.path.join('response_matrix', 'RM.txt')

# 读取响应矩阵数据
response_data = []
with open(rm_file, 'r') as f:
    for line in f:
        # 数据可能是逗号分隔的
        if ',' in line:
            values = [float(x) for x in line.strip().split(',')]
        else:
            values = [float(x) for x in line.strip().split()]
        response_data.append(values)

# 转换为numpy数组以便处理
response_matrix = np.array(response_data)

# 能量值 (从0.1到2.0 MeV)
energies = [0.1 * i for i in range(1, len(response_matrix) + 1)]
energy_labels = [f"{e:.1f} MeV" for e in energies]

# 闪烁体编号 (从1到20)
scintillator_ids = list(range(1, response_matrix.shape[1] + 1))

# 为第一幅图创建颜色循环 - 使用渐变色彩
colors1 = plt.cm.viridis(np.linspace(0, 1, len(response_matrix)))

# 第一幅图：每条曲线代表一个能量
plt.figure(figsize=(14, 10))

# 绘制每个能量的响应曲线
for i, response in enumerate(response_matrix):
    plt.plot(scintillator_ids, response, marker='o', linewidth=2, markersize=8, 
             color=colors1[i], label=energy_labels[i])

# 设置图表属性
plt.xlabel('Scintillator ID', fontsize=18)
plt.ylabel('Normalized Response', fontsize=18)
plt.title('Detector Response Matrix (by Energy)', fontsize=22)
plt.grid(True, linestyle='--', alpha=0.7)

# 设置x轴范围，确保从1开始
plt.xlim(0.5, len(scintillator_ids) + 0.5)
plt.xticks(scintillator_ids)

# 添加图例，并将其放在图表外部右侧
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# 调整布局，确保所有元素都能显示
plt.tight_layout()

# 保存图表
output_file = 'response_matrix/response_matrix_by_energy_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Response matrix plot (by energy) saved to {output_file}")

# 第二幅图：每条曲线代表一个探测器ID
plt.figure(figsize=(14, 10))

# 为第二幅图准备不同的线型、标记和颜色
linestyles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']

# 创建颜色映射
colors = plt.cm.tab20(np.linspace(0, 1, 20))  # tab20提供20种不同颜色
colors2 = np.vstack([colors, plt.cm.tab20b(np.linspace(0, 1, 20))])  # 备用颜色

# 转置矩阵，以便按探测器ID绘图
response_matrix_T = response_matrix.T

# 绘制每个探测器ID的响应曲线，使用不同的颜色、线型和标记组合
for i, detector_response in enumerate(response_matrix_T):
    linestyle = linestyles[i % len(linestyles)]
    marker = markers[i % len(markers)]
    color_idx = i % len(colors)
    
    plt.plot(energies, detector_response, 
             linestyle=linestyle, 
             marker=marker, 
             color=colors[color_idx],
             linewidth=2, 
             markersize=8, 
             label=f"ID {i+1}")

# 设置图表属性
plt.xlabel('Energy (MeV)', fontsize=18)
plt.ylabel('Normalized Response', fontsize=18)
plt.title('Detector Response Matrix (by Detector ID)', fontsize=22)
plt.grid(True, linestyle='--', alpha=0.7)

# 调整横轴刻度以显示更多能量值
plt.xticks(energies)

# 创建分组图例，使其更易读
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=2)

# 调整布局，确保所有元素都能显示
plt.tight_layout()

# 保存图表
output_file = 'response_matrix/response_matrix_by_detector_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Response matrix plot (by detector) saved to {output_file}")

# 显示图表 (可选)
plt.show() 