# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import concurrent.futures
from tqdm import tqdm # 进度条
import datetime
import threading
import csv
import pandas as pd
import shutil
import random  # 用于随机选择和生成随机噪声

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
iterations = 5000

# 是否保存对比图像
save_figure = False

# 是否保存重建数据（能谱、残差等）
save_data = False

# 采用CPU的最大核数，默认使用所有核
workers = 999

# 随机选择文件参数
MAX_FILE = 20  # 最大处理文件数，设为None或者很大的数表示处理所有文件

# 随机噪声参数
RANDOM_SEED = 42  # 随机数种子，确保结果可复现，设为None表示使用系统时间作为种子
RANDOM_SCALE = 0.01  # 随机噪声的幅度
# 随机噪声生成器函数，可以改成其他分布
def RANDOM_GENERATOR():
    return random.random() 

def add_random_noise(data, scale=RANDOM_SCALE, generator=RANDOM_GENERATOR):
    """
    Add random noise to data
    
    Parameters:
        data: numpy.ndarray, original data
        scale: float, noise amplitude
        generator: function, noise generator function
        
    Returns:
        numpy.ndarray, data with noise added
    """
    # Import numpy locally to ensure it's available in multiprocessing
    import numpy as np
    
    # Generate noise with the same shape as data
    noise_factors = np.array([1.0 + scale * generator() for _ in range(len(data))])
    # Apply noise
    return data * noise_factors

def process_data_file(data_file, output_dir, response_matrix, save_figure, save_data=False):
    """Process a single data file and save results"""

    
    try:
        # Ensure output directories exist
        reconstruction_figure_dir = os.path.join(output_dir, 'reconstruction_figure')
        reconstruction_data_dir = os.path.join(output_dir, 'reconstruction_data')
        os.makedirs(reconstruction_figure_dir, exist_ok=True)
        os.makedirs(reconstruction_data_dir, exist_ok=True)
        
        # Get current thread ID
        thread_id = threading.get_ident()
        
        # Read data file
        source_particle_energies, source_particle_counts, detector_ids, detector_response = load_data(data_file)
        
        # Convert data to numpy arrays
        source_particle_counts = np.array(source_particle_counts)
        detector_response = np.array(detector_response)
        
        # Scale detector response
        detector_response = detector_response * scale_matrix
        
        # Add random noise to detector response
        detector_response = add_random_noise(detector_response)
        
        
        # Create a progress bar for iteration progress (position=0 means at the top)
        file_name = os.path.splitext(os.path.basename(data_file))[0]
        inner_pbar = tqdm(
            range(iterations), 
            desc=f"MLEM {file_name}", 
            leave=False, 
            position=0, 
            ncols=80,
            bar_format='{l_bar}{bar:30}{r_bar}'
        )
        
        # Run MLEM algorithm
        reconstructed, detector_response_residuals, reconstructed_relative_source_residuals = mlem_algorithm(
            response_matrix, 
            detector_response,
            source_particle_counts,
            iterations=iterations,
            verbose=False,
            progress_bar=inner_pbar,
            early_stop=False,       # Enable early stopping
            tolerance=1e-6,        # Relative improvement tolerance
            no_improvement_count=20 # No improvement threshold count
        )
        
        # Close inner progress bar
        inner_pbar.close()
        
        # Calculate relative residuals for each energy point
        # If initial value is 0, set to 1 to avoid division by zero
        source_particle_counts[source_particle_counts == 0] = 1  # Replace 0 values with 1

        # use absolute value to avoid positive/negative cancellation
        recon_relative_residuals_per_energy = (abs(reconstructed - source_particle_counts)/source_particle_counts)
        
        # Get file name (without path and extension)
        file_name = os.path.splitext(os.path.basename(data_file))[0]
        
        # Plot and save reconstruction comparison
        if save_figure:
            figure_path = os.path.join(reconstruction_figure_dir, f"{file_name}.png")
            plotdata_path = os.path.join(reconstruction_figure_dir, f"{file_name}_plotdata.csv")
            plot_reconstruction_comparison(
                source_particle_energies,
                source_particle_counts,
                reconstructed,
                detector_response_residuals,
                reconstructed_relative_source_residuals,
                len(detector_response_residuals),  # Use actual iteration count
                save_figure_path=figure_path,
                save_data_path=plotdata_path if save_data else None  # Control data saving with parameter
            )
        
        # Save reconstruction data
        data_path = os.path.join(reconstruction_data_dir, f"{file_name}.txt")
        with open(data_path, 'w') as f:
            for energy, value in zip(source_particle_energies, reconstructed):
                f.write(f"{energy} {value}\n")
        
        # Save MLEM results to thread-specific CSV file
        csv_path = os.path.join(output_dir, f"MLEM_{thread_id}.csv")
        save_mlem_results_csv(csv_path, file_name, len(recon_relative_residuals_per_energy), recon_relative_residuals_per_energy)
        
        return f"Completed: {os.path.basename(data_file)}"
    except Exception as e:
        raise Exception(f"Error {os.path.basename(data_file)}: {str(e)}")

def merge_csv_files(output_dir):
    """Merge all thread CSV files into a master file, then delete the original files"""
    # Find all MLEM_*.csv files
    csv_files = glob.glob(os.path.join(output_dir, "MLEM_*.csv"))
    
    if not csv_files:
        print("No CSV result files found")
        return
    
    # Output merged file path
    merged_file = os.path.join(output_dir, "MLEM_results_merged.csv")
    
    # Merge all CSV files
    with open(merged_file, 'w', newline='') as outfile:
        # Create CSV writer
        writer = csv.writer(outfile)
        
        # Write header (from first file)
        with open(csv_files[0], 'r') as first_file:
            reader = csv.reader(first_file)
            header = next(reader)
            writer.writerow(header)
        
        # Process each file, write data rows (skip header)
        for file in csv_files:
            with open(file, 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(row)
    
    # Delete original CSV files
    for file in csv_files:
        os.remove(file)
    
    print(f"CSV files merged to: {merged_file}, original CSV files deleted")
    
    return merged_file

def analyze_residuals(merged_csv_file, output_dir):
    """Analyze merged CSV file, generate statistics and histograms"""
    # Read CSV file
    df = pd.read_csv(merged_csv_file)
    
    # Get energy point columns (eng1, eng2, ...)
    energy_columns = [col for col in df.columns if col.startswith('eng')]
    
    if not energy_columns:
        print("No energy point data found in CSV file")
        return
    
    # Create statistics result path
    stats_file = os.path.join(output_dir, "energy_residual_statistics.csv")
    
    # ---------------------------------
    # Modify here to save different data
    # ---------------------------------
    # Prepare statistics data
    stats_data = {
        'Statistic': ['Count', 'Mean', 'Variance', 'Std', 'Min', 'Max']
    }
    
    # 计算所有能量点合并后的总体统计，并在 eng1 之前插入 total 列
    total_values = df[energy_columns].to_numpy().ravel()
    stats_data['total'] = [
        len(total_values),            # Count (所有 eng* 值的总数量)
        np.mean(total_values),        # Mean
        np.var(total_values),         # Variance
        np.std(total_values),         # Std
        np.min(total_values),         # Min
        np.max(total_values)          # Max
    ]
    
    # Calculate statistics for each energy point
    means = []
    stds = []
    energy_labels = []
    
    for col in energy_columns:
        values = df[col].values
        mean_val = np.mean(values) 
        var_val = np.var(values)
        std_val = np.std(values)
        
        stats_data[col] = [
            len(values),                  # Count
            mean_val,                     # Mean
            var_val,                      # Variance
            std_val,                      # Standard deviation
            np.min(values),               # Minimum
            np.max(values)                # Maximum
        ]
        
        # Collect data for line plot
        means.append(mean_val)
        stds.append(std_val)
        energy_labels.append(col)
    
    # Save statistics data to CSV
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_file, index=False)
    
    print(f"Residual statistics saved to: {stats_file}")
    
    # 创建直方图
    plt.figure(figsize=(14, 8))
    
    # 为每个能量点绘制直方图
    for col in energy_columns:
        plt.hist(df[col].values, bins=20, alpha=0.5, label=col)
    
    plt.xlabel('Residual Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Source Recon-residual Distribution for Each Energy', fontsize=16)
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
    plt.title('Mean Source Recon-residual with Standard Deviation Error Bars for Each Energy Point', fontsize=16)
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
    # 设置随机数种子
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    
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
    all_data_files = glob.glob(os.path.join(data_dir, 'Data', '**', '*.txt'), recursive=True)
    
    if not all_data_files:
        raise Exception("未找到数据文件，请检查路径是否正确")
    
    # 随机选择指定数量的文件
    total_files = len(all_data_files)
    if MAX_FILE is not None and MAX_FILE < total_files:
        data_files = random.sample(all_data_files, MAX_FILE)
        file_percent = (MAX_FILE / total_files) * 100
        print(f"从{total_files}个文件中随机选择了{MAX_FILE}个进行处理，选择比例: {file_percent:.2f}%")
    else:
        data_files = all_data_files
        print(f"处理全部{total_files}个文件，选择比例: 100%")
    
    # 保存处理的参数信息和文件列表
    with open(os.path.join(output_dir, "process_info.txt"), "w") as f:
        f.write(f"处理时间: {current_datetime}\n")
        f.write(f"总文件数: {total_files}\n")
        f.write(f"处理文件数: {len(data_files)}\n")
        f.write(f"处理比例: {(len(data_files) / total_files) * 100:.2f}%\n")
        f.write(f"迭代次数: {iterations}\n")
        f.write(f"随机数种子: {RANDOM_SEED}\n")
        f.write(f"随机噪声幅度: {RANDOM_SCALE}\n")
        f.write("\n处理的文件列表:\n")
        for file in data_files:
            f.write(f"{file}\n")
    
    print(f"开始并行处理{len(data_files)}个数据文件...")
    
    # 获取CPU核心数
    max_workers = os.cpu_count()
    num_workers = min(max_workers, len(data_files), workers) 
    print(f"使用 {num_workers} 个CPU核心进行并行处理")
    
    # 使用ProcessPoolExecutor进行并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_data_file, data_file, output_dir, response_matrix, save_figure, save_data): data_file for data_file in data_files}
        
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
        print(f"- 处理参数信息: {os.path.join(output_dir, 'process_info.txt')}")

if __name__ == "__main__":
    main()  