import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
import csv
import tqdm

def load_response_matrix(response_matrix_file):
    """
    Load response matrix from file
    """
    response_matrix = []
    with open(response_matrix_file, 'r') as f:
        for line in f:
            if ',' in line:
                values = [float(x.strip()) for x in line.split(',')]
            else:
                values = [float(x) for x in line.split()]
            response_matrix.append(values)

    response_matrix = np.array(response_matrix)
    response_matrix = response_matrix.T
    return response_matrix

def mlem_algorithm(response_matrix, measure_data, source_data, iterations=100, verbose=False, progress_bar=None, early_stop=False, tolerance=1e-6, no_improvement_count=20):
    """
    MLEM algorithm implementation based on the standard formula
    
    λ_j^(n+1) = (λ_j^n / ∑_i c_ij) * ∑_i [c_ij * f_i / ∑_k(c_ik * λ_k^n)]
    
    Parameters:
        response_matrix: numpy.ndarray, response matrix (M x N)
            M rows (detectors), N columns (energy bins)
        measure_data: numpy.ndarray, detector measurements (M)
        source_data: numpy.ndarray, source data (N)
        iterations: int, number of iterations
        verbose: bool, whether to output detailed information
        progress_bar: tqdm progress bar, to show progress within iterations
        early_stop: bool, whether to stop early if convergence is reached
        tolerance: float, relative improvement tolerance for early stopping
        no_improvement_count: int, number of iterations with no improvement before stopping
        
    Returns:
        # lambda_history: history of results for each iteration (too many memory used, not used now)
        reconstructed: final reconstruction results
        detector_response_residuals: sum of squared residuals for each iteration
        reconstructed_relative_source_residuals: sum of squared residuals for each iteration
    """
    # Get shape of response matrix
    M, N = response_matrix.shape
    
    if verbose:
        print(f"Response matrix shape: {response_matrix.shape}")
        print(f"Measurement data shape: {measure_data.shape}")
    
    # Initialize lambda to match total counts
    total_counts = np.sum(measure_data)
    lambda_n = np.ones(N) * (total_counts / N)
    
    # Store sum of squared residuals for each iteration
    detector_response_residuals = np.zeros(iterations)
    reconstructed_relative_source_residuals = np.zeros(iterations)

    # Pre-calculate sum of response matrix columns (denominator part)
    sum_c_ij = np.sum(response_matrix, axis=0)  # ∑_i c_ij
    
    # Pre-transpose response matrix for backward projection
    response_matrix_T = response_matrix.T
    
    # Epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Variables for early stopping
    best_residual = float('inf')
    no_improvement_counter = 0
    
    # Start iteration
    start_time = time.time()
    
    # Create iteration range based on progress bar
    if progress_bar:
        iter_range = progress_bar
    else:
        iter_range = range(iterations)
    
    # Store final result
    reconstructed = lambda_n.copy()
    
    for n in iter_range:
        # Forward projection (predicted measurements)
        # detector_response = response_matrix × source
        detector_response_predicted = np.dot(response_matrix, lambda_n)  # ∑_k(c_ik * λ_k^n)
        
        # Calculate residual
        detector_response_residual = np.sum((detector_response_predicted - measure_data) ** 2)
        detector_response_residuals[n] = detector_response_residual

        # 如果source_data==0，那么设置为1，避免除以0错误
        source_data[source_data == 0] = 1
        reconstructed_relative_source_residual = np.sum(((lambda_n - source_data)/source_data) ** 2)
        reconstructed_relative_source_residuals[n] = reconstructed_relative_source_residual
        
        # Check for early stopping
        if early_stop:
            if detector_response_residual < best_residual * (1 - tolerance):
                # We have improvement
                best_residual = detector_response_residual
                no_improvement_counter = 0
                # Save current result as best
                reconstructed = lambda_n.copy()
            else:
                no_improvement_counter += 1
                
            # Stop if no improvement for several iterations
            if no_improvement_counter >= no_improvement_count:
                if verbose:
                    print(f"Early stopping at iteration {n+1}, no improvement for {no_improvement_count} iterations")
                # Truncate residuals array
                detector_response_residuals = detector_response_residuals[:n+1]
                break
        
        # Calculate correction factors
        ratio = measure_data / (detector_response_predicted + epsilon)  # f_i / ∑_k(c_ik * λ_k^n)
        
        # Backward projection and update
        # Back-projection: R^T × ratio
        correction = np.dot(response_matrix_T, ratio)  # ∑_i [c_ij * f_i / ∑_k(c_ik * λ_k^n)]
        
        # Update lambda
        lambda_n = lambda_n * correction / (sum_c_ij + epsilon)  # λ_j^(n+1) = (λ_j^n / ∑_i c_ij) * correction
        
        if verbose and (n + 1) % 100 == 0:
            print(f"Iteration {n+1}, Sum of squared residuals: {detector_response_residual:.6e}")
    
    # If we didn't stop early, use the final lambda_n
    if not early_stop or no_improvement_counter < no_improvement_count:
        reconstructed = lambda_n
    
    end_time = time.time()
    if verbose:
        print(f"MLEM algorithm completed, time used: {end_time - start_time:.2f} seconds")
    
    return reconstructed, detector_response_residuals, reconstructed_relative_source_residuals

def load_data(file_path):
    """
    Read data file with format:
    Line 1: Incident particle energies (comma-separated)
    Line 2: Corresponding particle counts (comma-separated)
    Lines 3-4: Empty
    Line 5: Detector IDs (comma-separated)
    Line 6: Detector responses (comma-separated)
    
    Parameters:
        file_path: str, file path
        
    Returns:
        particle_energies: incident particle energies
        particle_counts: corresponding particle counts
        detector_ids: detector IDs
        detector_response: detector responses
        
    Raises:
        ValueError: If the file format is incorrect
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Check if file has exactly 6 lines
            if len(lines) != 6:
                raise ValueError(f"File must have exactly 6 lines, but has {len(lines)}")
            
            # Line 1: Incident particle energies (comma-separated)
            particle_energies = [float(x.strip()) for x in lines[0].split(',')]
            
            # Line 2: Corresponding particle counts (comma-separated)
            particle_counts = [float(x.strip()) for x in lines[1].split(',')]
            
            # Lines 3-4 should be empty (or will be ignored)
            
            # Line 5: Detector IDs (comma-separated)
            detector_ids = [x.strip() for x in lines[4].split(',')]
            
            # Line 6: Detector responses (comma-separated)
            detector_response = [float(x.strip()) for x in lines[5].split(',')]
        
        return particle_energies, particle_counts, detector_ids, detector_response
    
    except Exception as e:
        raise ValueError(f"Error reading data file: {e}. File must be 6 lines with comma-separated values.")

def plot_reconstruction_comparison(energies, original, reconstructed, detector_response_residuals, reconstructed_relative_source_residuals, iterations, save_path=None):
    """
    Plot comparison between original source and reconstructed source
    
    Parameters:
        energies: list or array, energy values
        original: list or array, original source
        reconstructed: list or array, reconstructed source
        detector_response_residuals: list or array, residuals for each iteration
        reconstructed_relative_source_residuals: list or array, residuals for each iteration
        iterations: int, number of iterations
        save_path: str, path to save the figure
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 7))
    
    # Plot original vs reconstructed as line plots
    x_energies = np.array(energies)
    ax1.plot(x_energies, original, marker='o', linewidth=2, markersize=5, label='Original Distribution')
    ax1.plot(x_energies, reconstructed, marker='s', linewidth=2, markersize=5, label='Reconstructed')
    ax1.set_xlabel('Energy (MeV)', fontsize=14)
    ax1.set_ylabel('Particle Count', fontsize=14)
    ax1.set_title('Original vs Reconstructed Source', fontsize=16)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_yscale('log')
    
    # Plot detector response residuals
    ax2.plot(range(1, iterations + 1), detector_response_residuals, marker='o', markersize=3)
    ax2.set_xlabel('Iteration Number', fontsize=14)
    ax2.set_ylabel('Sum of Squared Residuals', fontsize=14)
    ax2.set_title('Detector Response Residuals', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_yscale('log')

    # Plot reconstructed relative source residuals
    ax3.plot(range(1, iterations + 1), reconstructed_relative_source_residuals, marker='o', markersize=3)
    ax3.set_xlabel('Iteration Number', fontsize=14)
    ax3.set_ylabel('Sum of Squared Residuals', fontsize=14)
    ax3.set_title('Recon- Relative Source Residuals', fontsize=16)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_yscale('log')
    
    # Plot relative difference between reconstructed and original
    rel_diff = (reconstructed - original) / original * 100
    ax4.plot(x_energies, rel_diff, marker='o', linewidth=2, color='red')
    ax4.axhline(y=0, color='black', linestyle='--')
    ax4.set_xlabel('Energy (MeV)', fontsize=14)
    ax4.set_ylabel('Relative Difference (%)', fontsize=14)
    ax4.set_title('Relative Diff- between Recon- and Source', fontsize=16)
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 关闭当前图形，释放内存
    plt.close('all')

    return fig, (ax1, ax2, ax3, ax4)


def save_mlem_results_csv(save_path, data_name, iterations, recon_relative_residuals_per_energy, create_new=False):
    """
    保存MLEM结果到CSV文件
    
    Parameters:
        save_path: str, CSV文件保存路径
        data_name: str, 原始数据文件名
        iterations: int, 迭代次数
        recon_relative_residuals_per_energy: numpy.ndarray, 每个能量点的残差
        create_new: bool, 是否创建新文件(True)或追加到现有文件(False)
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 获取当前时间
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 准备要写入的数据
    row_data = [current_time, data_name, iterations]
    row_data.extend(recon_relative_residuals_per_energy)
    
    # 检查文件是否存在
    file_exists = os.path.isfile(save_path)
    
    # 如果需要创建新文件，但文件已存在，则清空文件
    if create_new and file_exists:
        mode = 'w'
    else:
        # 追加模式
        mode = 'a'
    
    # 写入CSV文件
    with open(save_path, mode, newline='') as f:
        writer = csv.writer(f)
        
        # 如果需要创建新文件或文件不存在，写入表头
        if create_new or not file_exists:
            # 创建表头：Time, Data File, Iterations, eng1, eng2, ...
            headers = ['Time', 'Data File', 'Iterations']
            for i in range(len(recon_relative_residuals_per_energy)):
                headers.append(f'eng{i+1}')
            writer.writerow(headers)
        
        # 写入数据行
        writer.writerow(row_data)


def test_mlem():
    """
    Test MLEM algorithm
    1. Read response matrix
    2. Read or create test data
    3. Verify response matrix accuracy
    4. Use MLEM algorithm for reconstruction
    5. Calculate and plot results
    """

    
    def plot_source_detector_data(energies, source, detector_ids, response, title=None, save_path=None):
        """
        Plot source image and detector response
        
        Parameters:
            energies: list or array, energy values
            source: list or array, source intensity at each energy
            detector_ids: list or array, detector IDs
            response: list or array, detector response
            title: str, title of the plot
            save_path: str, path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot source image as line plot
        x_energies = np.array(energies)
        ax1.plot(x_energies, source, marker='o', linewidth=2, markersize=5)
        ax1.set_xlabel('Energy (MeV)', fontsize=14)
        ax1.set_ylabel('Particle Count', fontsize=14)
        ax1.set_title('Source Spectrum', fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot detector response as line plot
        detector_indices = np.array([int(id) for id in detector_ids])
        ax2.plot(detector_indices, response, marker='o', linewidth=2, markersize=5)
        ax2.set_xlabel('Detector ID', fontsize=14)
        ax2.set_ylabel('Response', fontsize=14)
        ax2.set_title('Detector Response', fontsize=16)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Use log scale for y-axis
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        
        # Ensure integer ticks for detector IDs
        ax2.set_xticks(detector_indices)
        
        if title:
            fig.suptitle(title, fontsize=18)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, (ax1, ax2)

    def create_test_data_file(file_path):
        """
        Create test data file with provided data
        
        Parameters:
            file_path: str, file path to save the test data
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Test data
        energies = "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0"
        source = "555970244, 256520761, 88924034, 53650561, 18216985, 12910774, 7602436, 4304839, 3016221, 1578175, 1432844, 1145311, 1002308, 717179, 572209, 429884, 430611, 287546, 286964, 143256"
        detector_ids = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20"
        response = "1.21165594e+08, 2.42644711e+07, 8.99585788e+06, 5.07272598e+06, 2.86282202e+06, 1.51821225e+06, 9.78645970e+05, 7.55388110e+05, 4.24862611e+05, 2.92131669e+05, 2.14215746e+05, 1.48203664e+05, 8.53865258e+04, 4.95396685e+04, 3.55880238e+04, 1.79152878e+04, 1.25906882e+04, 5.95954381e+03, 4.56650649e+03, 4.31204757e+03"
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(energies + '\n')
            f.write(source + '\n')
            f.write('\n\n')  # Empty lines 3-4
            f.write(detector_ids + '\n')
            f.write(response + '\n')
        
        print(f"Test data file created at: {file_path}")



    # Create test directory
    test_dir = os.path.join('tools_function', 'MLEM_test')
    os.makedirs(test_dir, exist_ok=True)
    
    # Set paths
    rm_file = os.path.join('response_matrix', 'RM.txt')
    test_data_file = os.path.join(test_dir, 'test_data.txt')
    
    # Create test data file
    if not os.path.exists(test_data_file):
        create_test_data_file(test_data_file)
    
    # Check if response matrix file exists
    if not os.path.exists(rm_file):
        print(f"Error: Response matrix file {rm_file} does not exist")
        return
    

    response_matrix = load_response_matrix(rm_file)

    # Read test data
    particle_energies, particle_counts, detector_ids, detector_response = load_data(test_data_file)
    
    # Convert to numpy arrays
    particle_counts = np.array(particle_counts)
    detector_response = np.array(detector_response)
    
    # Print data information
    print(f"Number of energy values: {len(particle_energies)}")
    print(f"Number of particle counts: {len(particle_counts)}")
    print(f"Number of detector IDs: {len(detector_ids)}")
    print(f"Number of detector responses: {len(detector_response)}")
    print(f"Response matrix shape: {response_matrix.shape}")
    
    # Check matrix dimensions
    print("\n=== Checking Matrix Dimensions ===")
    print(f"Response matrix shape: {response_matrix.shape}")
    print(f"Source vector shape: {particle_counts.shape}")
    print(f"Detector response vector shape: {detector_response.shape}")
    
    # Verify that matrix dimensions are consistent with expected multiplication
    # For R × S = D: R should be (M×N), S should be (N), D should be (M)
    expected_shape = (len(detector_response), len(particle_counts))
    if response_matrix.shape != expected_shape:
        print(f"WARNING: Response matrix shape is {response_matrix.shape}, but expected {expected_shape}")
        print("This may indicate the matrix is transposed from what we expect.")
        
        raise ValueError("Response matrix dimensions do not match expected dimensions")
    

    
    # Plot source and detector data
    plot_source_detector_data(
        particle_energies, 
        particle_counts, 
        detector_ids, 
        detector_response,
        title="Original Source and Detector Response",
        save_path=os.path.join(test_dir, 'original_data.png')
    )
    
    # Check if data shapes match
    if response_matrix.shape[0] != len(detector_response):
        print(f"Error: Response matrix rows ({response_matrix.shape[0]}) do not match detector response count ({len(detector_response)})")
        return
    
    if response_matrix.shape[1] != len(particle_counts):
        print(f"Error: Response matrix columns ({response_matrix.shape[1]}) do not match particle count length ({len(particle_counts)})")
        return
    
    # Set number of iterations
    iterations = 100
    
    # Run MLEM algorithm
    print("\n=== Running MLEM Algorithm ===")
    progress_bar = tqdm.tqdm(total=iterations)
    reconstructed, residuals = mlem_algorithm(
        response_matrix, 
        detector_response,
        iterations=iterations,
        verbose=True,
        progress_bar=progress_bar
    )
    
    
    # Calculate difference from true values
    final_residual = np.sum(((reconstructed - particle_counts)/particle_counts) ** 2) / len(particle_counts)
    
    print(f"\nMELM algorithm - Final sum of squared residuals: {final_residual:.6e}")
    
    # Plot reconstruction comparisons
    plot_reconstruction_comparison(
        particle_energies,
        particle_counts,
        reconstructed,
        residuals,
        iterations,
        save_path=os.path.join(test_dir, 'reconstruction_results.png')
    )
    
    # 关闭当前图形，释放内存
    plt.close('all')

    print(f"All results saved to {test_dir}")

    # Save MLEM results to CSV
    save_mlem_results_csv(os.path.join(test_dir, 'MLEM_results.csv'), 'test_data', iterations, residuals)


if __name__ == "__main__":
    test_mlem()