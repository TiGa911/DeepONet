# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import re

def parse_timing_file(file_path):
    """
    解析单个时间统计文件，提取各模块时间
    """
    timing_data = {
        'data_reading': 0,
        'te_fitting': 0,
        'ne_fitting': 0,
        'ti_fitting': 0,
        'lhcd_model': 0,
        'ecrh_setup': 0,
        'onetwo_run': 0,
        'total': 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式提取时间数据
        patterns = {
            'data_reading': r'数据读取\s+([\d.]+)',
            'te_fitting': r'Te剖面拟合\s+([\d.]+)',
            'ne_fitting': r'ne剖面拟合\s+([\d.]+)',
            'ti_fitting': r'Ti剖面拟合\s+([\d.]+)',
            'lhcd_model': r'低杂波模型\s+([\d.]+)',
            'ecrh_setup': r'ECRH设置\s+([\d.]+)',
            'onetwo_run': r'ONETWO运行\s+([\d.]+)',
            'total': r'总计\s+([\d.]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                # 提取数字部分，去掉可能的's'单位
                time_str = match.group(1).replace('s', '').strip()
                try:
                    timing_data[key] = float(time_str)
                except ValueError:
                    print(f"警告: 无法转换 '{time_str}' 为数字，在文件 {file_path}")
                    timing_data[key] = 0
    
    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
        return None
    
    # 检查是否有有效数据
    if timing_data['total'] == 0:
        print(f"警告: 文件 {file_path} 中没有有效的时间数据")
        return None
    
    return timing_data

def calculate_average_times(shot_number, base_dir="results"):
    """
    计算指定炮号的所有时间点的平均计算时间
    """
    # 查找所有时间统计文件
    pattern = os.path.join(base_dir, str(shot_number), "*", "timing_detailed.txt")
    timing_files = glob.glob(pattern)
    
    if not timing_files:
        print(f"未找到炮号 {shot_number} 的时间统计文件")
        return None
    
    print(f"找到 {len(timing_files)} 个时间统计文件")
    
    # 解析所有文件
    all_timing_data = []
    valid_files = 0
    
    for file_path in timing_files:
        timing_data = parse_timing_file(file_path)
        if timing_data and timing_data['total'] > 0:  # 只统计有效的文件
            all_timing_data.append(timing_data)
            valid_files += 1
    
    if valid_files == 0:
        print("没有找到有效的时间数据")
        return None
    
    print(f"成功解析 {valid_files} 个有效时间文件")
    
    # 计算平均值和标准差
    df = pd.DataFrame(all_timing_data)
    
    # 计算ECRH总时间（ECRH设置 + ONETWO运行）
    df['ecrh_total'] = df['ecrh_setup'] + df['onetwo_run']
    
    # 计算平均值和标准差
    stats = {
        'count': valid_files,
        'data_reading': {'mean': df['data_reading'].mean(), 'std': df['data_reading'].std()},
        'te_fitting': {'mean': df['te_fitting'].mean(), 'std': df['te_fitting'].std()},
        'ne_fitting': {'mean': df['ne_fitting'].mean(), 'std': df['ne_fitting'].std()},
        'ti_fitting': {'mean': df['ti_fitting'].mean(), 'std': df['ti_fitting'].std()},
        'lhcd_model': {'mean': df['lhcd_model'].mean(), 'std': df['lhcd_model'].std()},
        'ecrh_setup': {'mean': df['ecrh_setup'].mean(), 'std': df['ecrh_setup'].std()},
        'onetwo_run': {'mean': df['onetwo_run'].mean(), 'std': df['onetwo_run'].std()},
        'ecrh_total': {'mean': df['ecrh_total'].mean(), 'std': df['ecrh_total'].std()},
        'total': {'mean': df['total'].mean(), 'std': df['total'].std()}
    }
    
    return stats, df

def generate_report(shot_number, stats, df, output_dir="results"):
    """
    生成详细的统计报告
    """
    report_file = os.path.join(output_dir, f"timing_analysis_shot{shot_number}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"              EAST Plasma Simulation Timing Analysis Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Shot: {shot_number}\n")
        f.write(f"Number of time points analyzed: {stats['count']}\n")
        f.write(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 70 + "\n")
        f.write("Average Computation Time Statistics by Module:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Module':<20} {'Mean Time(s)':<15} {'Std Dev(s)':<15} {'Percentage(%)':<10}\n")
        f.write("-" * 70 + "\n")
        
        total_mean = stats['total']['mean']
        
        modules = [
            ('Data Reading', stats['data_reading']),
            ('Te Profile Fitting', stats['te_fitting']),
            ('ne Profile Fitting', stats['ne_fitting']),
            ('Ti Profile Fitting', stats['ti_fitting']),
            ('LHCD Model', stats['lhcd_model']),
            ('ECRH Setup', stats['ecrh_setup']),
            ('ONETWO Execution', stats['onetwo_run']),
            ('ECRH Total', stats['ecrh_total']),
            ('Total', stats['total'])
        ]
        
        for name, data in modules:
            mean = data['mean']
            std = data['std']
            percentage = (mean / total_mean * 100) if total_mean > 0 else 0
            f.write(f"{name:<20} {mean:<15.3f} {std:<15.3f} {percentage:<10.1f}\n")
        
        f.write("-" * 70 + "\n")
        
        # 性能分析
        f.write("\nPerformance Analysis:\n")
        f.write("-" * 70 + "\n")
        
        # 最耗时的模块
        module_times = {
            'Data Reading': stats['data_reading']['mean'],
            'Te Fitting': stats['te_fitting']['mean'],
            'ne Fitting': stats['ne_fitting']['mean'],
            'Ti Fitting': stats['ti_fitting']['mean'],
            'LHCD Model': stats['lhcd_model']['mean'],
            'ECRH Setup': stats['ecrh_setup']['mean'],
            'ONETWO Execution': stats['onetwo_run']['mean']
        }
        
        max_module = max(module_times.items(), key=lambda x: x[1])
        f.write(f"Most Time-Consuming Module: {max_module[0]} ({max_module[1]:.3f}s, {max_module[1]/total_mean*100:.1f}%)\n")
        
        # ECRH相关时间分析
        ecrh_percentage = stats['ecrh_total']['mean'] / total_mean * 100
        f.write(f"ECRH-Related Computation Percentage: {ecrh_percentage:.1f}%\n")
        
        # 拟合总时间
        fitting_total = (stats['te_fitting']['mean'] + 
                        stats['ne_fitting']['mean'] + 
                        stats['ti_fitting']['mean'])
        fitting_percentage = fitting_total / total_mean * 100
        f.write(f"Total Profile Fitting Time: {fitting_total:.3f}s ({fitting_percentage:.1f}%)\n")
        
        # 计算效率分析
        f.write(f"Average Total Time per Time Point: {total_mean:.3f}s\n")
        if stats['count'] > 1:
            f.write(f"Time Stability (Std Dev): {stats['total']['std']:.3f}s\n")
        
        f.write("-" * 70 + "\n")
        
        # 详细数据摘要
        f.write("\nDetailed Data Summary:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Data Reading Time Range: {df['data_reading'].min():.3f}s - {df['data_reading'].max():.3f}s\n")
        f.write(f"LHCD Model Time Range: {df['lhcd_model'].min():.3f}s - {df['lhcd_model'].max():.3f}s\n")
        f.write(f"ECRH Total Time Range: {df['ecrh_total'].min():.3f}s - {df['ecrh_total'].max():.3f}s\n")
        f.write(f"Total Time Range: {df['total'].min():.3f}s - {df['total'].max():.3f}s\n")
        
        # 时间分布分析
        f.write("\nTime Distribution Analysis:\n")
        f.write("-" * 70 + "\n")
        f.write(f"25th Percentile: {df['total'].quantile(0.25):.3f}s\n")
        f.write(f"Median: {df['total'].median():.3f}s\n")
        f.write(f"75th Percentile: {df['total'].quantile(0.75):.3f}s\n")
        
        # 异常值检测
        Q1 = df['total'].quantile(0.25)
        Q3 = df['total'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[df['total'] > upper_bound]
        
        if len(outliers) > 0:
            f.write(f"Detected {len(outliers)} outlier time points (> {upper_bound:.3f}s)\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"Analysis report saved to: {report_file}")
    return report_file

def create_visualization(shot_number, df, output_dir="results"):
    """
    创建简单的可视化图表（可选）
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        
        # 使用默认字体，避免中文字体问题
        plt.rcParams.update({'font.size': 10})
        
        # 创建时间分布图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 各模块时间占比饼图
        module_times = [
            df['data_reading'].mean(),
            df['te_fitting'].mean(),
            df['ne_fitting'].mean(),
            df['ti_fitting'].mean(),
            df['lhcd_model'].mean(),
            df['ecrh_setup'].mean(),
            df['onetwo_run'].mean()
        ]
        module_labels = ['Data Reading', 'Te Fitting', 'ne Fitting', 'Ti Fitting', 
                        'LHCD Model', 'ECRH Setup', 'ONETWO Execution']
        
        axes[0, 0].pie(module_times, labels=module_labels, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Time Distribution by Module')
        
        # 2. 总时间分布直方图
        axes[0, 1].hist(df['total'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].axvline(df['total'].mean(), color='red', linestyle='--', label=f'Mean: {df["total"].mean():.2f}s')
        axes[0, 1].set_xlabel('Total Time (s)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Total Time Distribution')
        axes[0, 1].legend()
        
        # 3. ECRH相关时间分析
        ecrh_components = ['ECRH Setup', 'ONETWO Execution']
        ecrh_times = [df['ecrh_setup'].mean(), df['onetwo_run'].mean()]
        
        axes[1, 0].bar(ecrh_components, ecrh_times, color=['lightcoral', 'lightgreen'])
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].set_title('ECRH-Related Time Analysis')
        
        # 4. 主要模块时间箱线图 - 修复labels参数警告
        main_modules = ['Data Reading', 'LHCD Model', 'ECRH Total']
        main_data = [df['data_reading'], df['lhcd_model'], df['ecrh_total']]
        
        # 检查matplotlib版本并选择合适的参数
        matplotlib_version = matplotlib.__version__
        if tuple(map(int, matplotlib_version.split('.')[:2])) >= (3, 9):
            axes[1, 1].boxplot(main_data, tick_labels=main_modules)
        else:
            axes[1, 1].boxplot(main_data, labels=main_modules)
        
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].set_title('Main Modules Time Distribution')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"timing_analysis_shot{shot_number}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization chart saved to: {plot_file}")
        
    except ImportError:
        print("Matplotlib not installed, skipping visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")

def main():
    """
    主函数
    """
    print("EAST Plasma Simulation Timing Analysis Tool")
    print("=" * 50)
    
    try:
        shot_number = int(input("Enter shot number: "))
        base_dir = input("Enter results directory (default 'results'): ").strip()
        if not base_dir:
            base_dir = "results"
        
        # 计算平均时间
        result = calculate_average_times(shot_number, base_dir)
        
        if result is None:
            return
        
        stats, df = result
        
        # 生成报告
        report_file = generate_report(shot_number, stats, df, base_dir)
        
        # 创建可视化图表
        create_visualization(shot_number, df, base_dir)
        
        # 在控制台输出简要结果
        print("\nSummary Results:")
        print(f"- Number of time points analyzed: {stats['count']}")
        print(f"- Average total time: {stats['total']['mean']:.3f}s")
        print(f"- Average ECRH total time: {stats['ecrh_total']['mean']:.3f}s")
        print(f"- ECRH percentage: {stats['ecrh_total']['mean']/stats['total']['mean']*100:.1f}%")
        print(f"- Average LHCD model time: {stats['lhcd_model']['mean']:.3f}s")
        
    except ValueError:
        print("Please enter a valid shot number (integer)")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()