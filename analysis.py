# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import concurrent.futures
from tqdm import tqdm # 进度条
from tqdm.auto import trange  # 进度条
import datetime
import threading
import csv
import pandas as pd
import shutil

# 将MLEM模块所在目录添加到系统路径
sys.path.append('tools_function')
from tools_function.MLEM import load_data, load_response_matrix, mlem_algorithm, plot_reconstruction_comparison, save_mlem_results_csv

# ================================================
# ================ 脚本配置参数 ===================
# ================================================

# 设置文件路径
data_dir = r'数据集'  # 数据目录
response_matrix_file = r'response_matrix\RM.txt'  # 响应矩阵文件路径

# 对探测器响应进行缩放因子 (需要和计算响应矩阵时的缩放因子一致)
scale_matrix = [0.92472407, 1.27042858, 1.13919637, 1.05919509, 0.79361118, 0.79359671, 0.73485017, 1.21970569, 1.06066901, 1.12484355, 0.7123507, 1.28194591, 1.19946558, 0.82740347, 0.80909498, 0.81004271, 0.88254535, 1.01485386, 0.95916701, 0.87473748]

# 设置MLEM参数
iterations = 1000000

# 是否保存对比图像
save_figure = False

# 采用CPU的最大核数，默认使用所有核
workers = 999

def process_data_file(data_file, output_dir, response_matrix, save_figure):
    """处理单个数据文件并保存结果"""
    try:
        # 确保输出目录存在
        reconstruction_figure_dir = os.path.join(output_dir, 'reconstruction_figure')
        reconstruction_data_dir = os.path.join(output_dir, 'reconstruction_data')
        os.makedirs(reconstruction_figure_dir, exist_ok=True)
        os.makedirs(reconstruction_data_dir, exist_ok=True)
        
        # 获取当前线程ID
        thread_id = threading.get_ident()
        
        # 读取数据文件
        particle_energies, particle_counts, detector_ids, detector_response = load_data(data_file)
        
        # 将数据转换为numpy数组
        particle_counts = np.array(particle_counts)
        detector_response = np.array(detector_response)
        
        # 对探测器响应进行缩放
        detector_response = detector_response * scale_matrix
        
        # 创建一个进度条用于显示迭代进度（position=1表示在主进度条上方）
        file_name = os.path.splitext(os.path.basename(data_file))[0]
        inner_pbar = tqdm(
            range(iterations), 
            desc=f"MLEM {file_name}", 
            leave=False, 
            position=0, 
            ncols=80,
            bar_format='{l_bar}{bar:30}{r_bar}'
        )
        
        # 运行MLEM算法
        reconstructed, residuals = mlem_algorithm(
            response_matrix, 
            detector_response,
            iterations=iterations,
            verbose=False,
            progress_bar=inner_pbar,
            early_stop=False,       # 启用提前停止条件
            tolerance=1e-6,        # 相对改进容差
            no_improvement_count=20 # 无改进次数阈值
        )
        
        # 关闭内部进度条
        inner_pbar.close()
        
        # 计算每个能量点的相对残差
        residuals_per_energy = ((reconstructed - particle_counts)/particle_counts) ** 2
        
        # 获取文件名（不包含路径和扩展名）
        file_name = os.path.splitext(os.path.basename(data_file))[0]
        
        # 绘制重建结果对比图并保存
        if save_figure:
            figure_path = os.path.join(reconstruction_figure_dir, f"{file_name}.png")
            plot_reconstruction_comparison(
                particle_energies,
                particle_counts,
                reconstructed,
                residuals,
                len(residuals),  # 使用实际迭代次数
                save_path=figure_path
            )
        
        # 保存重建数据
        data_path = os.path.join(reconstruction_data_dir, f"{file_name}.txt")
        with open(data_path, 'w') as f:
            for energy, value in zip(particle_energies, reconstructed):
                f.write(f"{energy} {value}\n")
        
        # 保存MLEM结果到线程特定的CSV文件
        csv_path = os.path.join(output_dir, f"MLEM_{thread_id}.csv")
        save_mlem_results_csv(csv_path, file_name, len(residuals), residuals_per_energy)
        
        return f"完成: {os.path.basename(data_file)}"
    except Exception as e:
        return f"错误 {os.path.basename(data_file)}: {str(e)}"

def merge_csv_files(output_dir):
    """合并所有线程的CSV文件到一个主文件，然后删除原始文件"""
    # 查找所有MLEM_*.csv文件
    csv_files = glob.glob(os.path.join(output_dir, "MLEM_*.csv"))
    
    if not csv_files:
        print("未找到CSV结果文件")
        return
    
    # 输出合并文件路径
    merged_file = os.path.join(output_dir, "MLEM_results_merged.csv")
    
    # 合并所有CSV文件
    with open(merged_file, 'w', newline='') as outfile:
        # 创建CSV写入器
        writer = csv.writer(outfile)
        
        # 写入表头（从第一个文件获取）
        with open(csv_files[0], 'r') as first_file:
            reader = csv.reader(first_file)
            header = next(reader)
            writer.writerow(header)
        
        # 逐个处理文件，写入数据行（跳过表头）
        for file in csv_files:
            with open(file, 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # 跳过表头
                for row in reader:
                    writer.writerow(row)
    
    # 删除原始CSV文件
    for file in csv_files:
        os.remove(file)
    
    print(f"CSV文件已合并到: {merged_file}，原始CSV文件已删除")
    
    return merged_file

def analyze_residuals(merged_csv_file, output_dir):
    """分析合并后的CSV文件，生成统计数据和直方图"""
    # 读取CSV文件
    df = pd.read_csv(merged_csv_file)
    
    # 获取能量点列（eng1, eng2, ...）
    energy_columns = [col for col in df.columns if col.startswith('eng')]
    
    if not energy_columns:
        print("CSV文件中未找到能量点数据")
        return
    
    # 创建统计结果的路径
    stats_file = os.path.join(output_dir, "energy_residual_statistics.csv")
    
    # ---------------------------------
    # 在这里修改，需要保存哪些数据
    # ---------------------------------
    # 准备统计数据
    stats_data = {
        'Statistic': ['Count', 'Mean', 'Variance', 'Std', 'Min', 'Max']
    }
    
    # 为每个能量点计算统计数据
    means = []
    stds = []
    energy_labels = []
    
    for col in energy_columns:
        values = df[col].values
        mean_val = np.mean(values)
        var_val = np.var(values)
        std_val = np.std(values)
        
        stats_data[col] = [
            len(values),                  # 数量
            mean_val,                     # 均值
            var_val,                      # 方差
            std_val,                      # 标准差
            np.min(values),               # 最小值
            np.max(values)                # 最大值
        ]
        
        # 收集用于绘制折线图的数据
        means.append(mean_val)
        stds.append(std_val)
        energy_labels.append(col)
    
    # 保存统计数据到CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_file, index=False)
    
    print(f"残差统计数据已保存到: {stats_file}")
    
    # 创建直方图
    plt.figure(figsize=(14, 8))
    
    # 为每个能量点绘制直方图
    for col in energy_columns:
        plt.hist(df[col].values, bins=20, alpha=0.5, label=col)
    
    plt.xlabel('Residual Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Residual Distribution for Each Energy Point', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存直方图
    hist_path = os.path.join(output_dir, "energy_residual_histogram.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建折线图（均值和标准差）
    plt.figure(figsize=(14, 8))
    
    # 注意：这里使用x轴为1到N，而非实际的能量值（因为我们只有eng1,eng2...这样的标签）
    x_values = np.arange(1, len(means) + 1)
    
    # 绘制折线图，带误差棒
    plt.errorbar(x_values, means, yerr=stds, fmt='o-', ecolor='red', capsize=5, capthick=2, linewidth=2)
    
    plt.xlabel('Energy Point Index', fontsize=14)
    plt.ylabel('Mean Residual Value', fontsize=14)
    plt.title('Mean Residual with Standard Deviation Error Bars for Each Energy Point', fontsize=16)
    plt.xticks(x_values, energy_labels, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存折线图
    errorbar_path = os.path.join(output_dir, "energy_residual_errorbar.png")
    plt.tight_layout()
    plt.savefig(errorbar_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"残差分布直方图已保存到: {hist_path}")
    print(f"残差均值与标准差折线图已保存到: {errorbar_path}")
    
    return stats_file, hist_path, errorbar_path

def main():
    """主函数，处理所有数据文件"""
    # 在这里定义输出目录
    current_datetime = datetime.datetime.now().strftime('%y%m%d_%H%M%S') 
    output_dir = f'output_{current_datetime}'  # 结果保存目录
    
    # 创建输出目录结构
    os.makedirs(output_dir, exist_ok=True)
    reconstruction_figure_dir = os.path.join(output_dir, 'reconstruction_figure')
    reconstruction_data_dir = os.path.join(output_dir, 'reconstruction_data')
    os.makedirs(reconstruction_figure_dir, exist_ok=True)
    os.makedirs(reconstruction_data_dir, exist_ok=True)
    
    # 读取响应矩阵
    response_matrix = load_response_matrix(response_matrix_file)
    
    # 搜索数据集/Data/目录下的所有txt文件
    data_files = glob.glob(os.path.join(data_dir, 'Data', '**', '*.txt'), recursive=True)
    
    if not data_files:
        raise Exception("未找到数据文件，请检查路径是否正确")

    
    print(f"找到 {len(data_files)} 个数据文件，开始并行处理...")
    
    # 获取CPU核心数
    max_workers = os.cpu_count()
    num_workers = min(max_workers, len(data_files), workers) 
    print(f"使用 {num_workers} 个CPU核心进行并行处理")
    
    # 使用ProcessPoolExecutor进行并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_data_file, data_file, output_dir, response_matrix, save_figure): data_file for data_file in data_files}
        
        # 创建进度条（position=1表示在内部进度条下方）
        with tqdm(total=len(data_files), desc="文件处理进度", ncols=100, leave=True, position=1) as pbar:
            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_file):
                data_file = future_to_file[future]
                try:
                    result = future.result()
                    # 更新进度条描述，显示最近完成的文件
                    pbar.set_description(f"完成: {os.path.basename(data_file)}")
                except Exception as exc:
                    print(f"处理文件 {data_file} 时出错: {exc}")
                finally:
                    # 更新进度条
                    pbar.update(1)
    
    # 合并所有CSV文件
    merged_file = merge_csv_files(output_dir)
    
    if merged_file:
        # 分析残差数据
        stats_file, hist_path, errorbar_path = analyze_residuals(merged_file, output_dir)
        
        print(f"分析完成，结果保存至:")
        print(f"- 重建图像: {reconstruction_figure_dir}")
        print(f"- 重建数据: {reconstruction_data_dir}")
        print(f"- 合并CSV结果: {merged_file}")
        print(f"- 残差统计数据: {stats_file}")
        print(f"- 残差分布直方图: {hist_path}")
        print(f"- 残差均值与标准差折线图: {errorbar_path}")

if __name__ == "__main__":
    main()  