import os
import numpy as np
import glob

# 定义路径
DATA_PATH = os.path.join('数据集', 'JiTai_Data')
OUTPUT_PATH = 'response_matrix'

# 设置探测器响应的缩放因子
scale_matrix = [0.92472407, 1.27042858, 1.13919637, 1.05919509, 0.79361118, 0.79359671, 0.73485017, 1.21970569, 1.06066901, 1.12484355, 0.7123507, 1.28194591, 1.19946558, 0.82740347, 0.80909498, 0.81004271, 0.88254535, 1.01485386, 0.95916701, 0.87473748];

# 创建输出目录
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# 处理各个能量下的数据文件
energy_folders = sorted([f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))], 
                        key=lambda x: float(x.replace('MeV', '')))

print(f"找到以下能量文件夹: {energy_folders}")

final_response = []  # 用于存储最终的响应矩阵数据

for energy_folder in energy_folders:
    print(f"处理 {energy_folder} 中的数据...")
    
    # 获取当前处理的能量值（从文件夹名称中提取）
    current_energy = float(energy_folder.replace('MeV', ''))
    
    # 这个能量文件夹的路径
    folder_path = os.path.join(DATA_PATH, energy_folder)
    
    # 查找所有数据文件
    data_files = glob.glob(os.path.join(folder_path, 'test_run_event_*.txt'))
    
    if not data_files:
        print(f"警告: 在 {energy_folder} 中没有找到数据文件")
        continue
    
    print(f"在 {energy_folder} 中找到 {len(data_files)} 个数据文件")
    
    # 用于累积结果
    total_particles = 0  # 入射粒子总数
    detector_ids = None  # 探测器ID列表
    deposit_energy = None  # 沉积能量
    
    # 处理每个数据文件
    processed_files = 0
    for data_file in data_files:
        try:
            with open(data_file, 'r') as f:
                lines = f.readlines()
                
                # 确保文件至少有6行
                if len(lines) < 6:
                    print(f"警告: 文件 {data_file} 格式不正确，跳过")
                    continue
                
                # 提取数据
                try:
                    # 第一行包含所有能量值，逗号分隔
                    energies = [float(e) for e in lines[0].split(',')]
                    
                    # 第二行包含对应能量的粒子数量
                    particles = [int(p) for p in lines[1].split(',')]
                    
                    # 找到当前能量在列表中的索引
                    energy_index = -1
                    for i, e in enumerate(energies):
                        if abs(e - current_energy) < 0.01:  # 允许一点小数误差
                            energy_index = i
                            break
                    
                    if energy_index == -1:
                        print(f"警告: 文件 {data_file} 中未找到与文件夹 {energy_folder} 匹配的能量值")
                        print(f"文件中的能量值: {energies}")
                        print(f"当前处理的能量值: {current_energy}")
                        continue
                    
                    # 获取对应能量的粒子数量
                    file_particles = particles[energy_index]
                    
                except (ValueError, IndexError) as e:
                    print(f"警告: 解析文件 {data_file} 的能量或粒子数时出错: {e}")
                    print(f"第一行内容: '{lines[0]}'")
                    print(f"第二行内容: '{lines[1]}'")
                    continue
                
                # 第五行是探测器ID，逗号分隔
                file_detector_ids = lines[4].split(',')
                
                # 第六行是沉积能量，逗号分隔
                try:
                    file_deposit = [float(x) for x in lines[5].split(',')]
                    # print("对探测器响应进行缩放")
                    file_deposit = [x * scale_matrix[i] for i, x in enumerate(file_deposit)]
                except ValueError as e:
                    print(f"警告: 解析文件 {data_file} 的沉积能量时出错: {e}")
                    print(f"第六行内容: '{lines[5]}'")
                    continue
                
                # 初始化或更新探测器ID
                if detector_ids is None:
                    detector_ids = file_detector_ids
                elif len(detector_ids) != len(file_detector_ids):
                    print(f"警告: 文件 {data_file} 中的探测器ID数量与其他文件不一致")
                    print(f"已有ID数量: {len(detector_ids)}, 当前文件ID数量: {len(file_detector_ids)}")
                    continue
                
                # 累加粒子数和沉积能量
                total_particles += file_particles
                
                if deposit_energy is None:
                    deposit_energy = file_deposit
                else:
                    # 确保两个列表长度相同
                    if len(deposit_energy) != len(file_deposit):
                        print(f"警告: 文件 {data_file} 中的沉积能量长度与其他文件不一致")
                        print(f"已有长度: {len(deposit_energy)}, 当前文件长度: {len(file_deposit)}")
                        continue
                    
                    deposit_energy = [a + b for a, b in zip(deposit_energy, file_deposit)]
                
                processed_files += 1
                    
        except Exception as e:
            print(f"处理文件 {data_file} 时出错: {e}")
    
    print(f"成功处理了 {processed_files}/{len(data_files)} 个文件")
    
    # 如果成功处理了数据，保存该能量的响应结果
    if total_particles > 0 and deposit_energy is not None:
        # 能量响应文件名
        response_file = os.path.join(OUTPUT_PATH, f"{energy_folder}_response.txt")
        
        with open(response_file, 'w') as f:
            f.write(f"{current_energy}\n")  # 第一行：入射粒子能量
            f.write(f"{total_particles}\n")  # 第二行：入射粒子数量
            f.write("\n")  # 第三行：空
            f.write("\n")  # 第四行：空
            
            # 使用逗号分隔符输出
            f.write(",".join(detector_ids) + "\n")  # 第五行：探测器ID
            f.write(",".join([str(e) for e in deposit_energy]) + "\n")  # 第六行：沉积能量
        
        print(f"已保存 {energy_folder} 的响应结果到 {response_file}")
        
        # 计算归一化的响应并添加到最终矩阵
        normalized_response = [e / total_particles for e in deposit_energy]
        final_response.append(normalized_response)
    else:
        print(f"警告: 未能为 {energy_folder} 生成有效的响应数据")
        if total_particles == 0:
            print("  - 粒子总数为零")
        if deposit_energy is None:
            print("  - 未找到有效的沉积能量数据")

# 保存响应矩阵
if final_response:
    rm_file = os.path.join(OUTPUT_PATH, "RM.txt")
    with open(rm_file, 'w') as f:
        for response in final_response:
            # 使用逗号分隔符输出
            f.write(",".join([str(e) for e in response]) + "\n")
    
    print(f"已保存响应矩阵到 {rm_file}")
    print(f"处理完成! 共处理了 {len(final_response)}/{len(energy_folders)} 个能量文件夹")
else:
    print("警告: 没有处理到任何有效的数据文件!")
