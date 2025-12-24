import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import functools
import multiprocessing
from multiprocessing import Pool, cpu_count
from scipy.ndimage import median_filter

# Py-ART库管理
import os

# 设置环境变量以禁用Py-ART欢迎信息
os.environ["PYART_QUIET"] = "1"

# Windows系统上使用multiprocessing库的必要步骤
if __name__ != "__main__":
    multiprocessing.freeze_support()

# 全局Py-ART库实例
_pyart = None

def get_pyart():
    """
    获取或加载pyart库实例，确保仅加载一次
    
    返回:
        module: pyart库模块
    """
    global _pyart
    
    # 如果已经加载，直接返回
    if _pyart is not None:
        return _pyart
    
    # 导入Py-ART（由于设置了PYART_QUIET环境变量，不会打印欢迎信息）
    import pyart
    _pyart = pyart
    return _pyart

# 全局进程池管理
_global_pool = None


def get_global_pool():
    """
    获取或创建全局进程池实例，确保仅创建一次
    
    返回:
        Pool: 全局进程池实例
    """
    global _global_pool
    if _global_pool is None:
        # 根据CPU核心数创建进程池
        num_processes = max(1, cpu_count() - 1)
        _global_pool = Pool(processes=num_processes)
        print(f"创建全局进程池，使用 {num_processes} 个进程")
    return _global_pool


def close_global_pool():
    """
    关闭全局进程池
    """
    global _global_pool
    if _global_pool is not None:
        _global_pool.close()
        _global_pool.join()
        _global_pool = None
        print("已关闭全局进程池")

# 解决matplotlib新版本兼容性问题
# 为避免Windows系统上multiprocessing创建的子进程重复显示消息，移除了打印输出
if not hasattr(matplotlib.cm, 'register_cmap'):
    # 为较新版本的matplotlib添加兼容层
    def register_cmap(**kwargs):
        pass
    matplotlib.cm.register_cmap = register_cmap




def timing_decorator(func):
    """
    性能计时装饰器，用于测量函数执行时间
    
    参数:
        func (callable): 要测量的函数
    
    返回:
        callable: 包装后的函数，包含执行时间信息
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result
    return wrapper





def check_reflectivity(radar):
    """
    检查反射率因子的有效性范围、数据完整性及极端天气特征
    
    参数:
        radar (Radar): 雷达数据对象
    
    返回:
        dict: 反射率检查结果，包含状态和详细信息
    """
    results = {
        'parameter': 'reflectivity',
        'status': 'pass',
        'issues': [],
        'stats': {}
    }
    
    if 'reflectivity' not in radar.fields:
        results['status'] = 'fail'
        results['issues'].append('反射率因子字段不存在')
        return results
    
    ref_data = radar.fields['reflectivity']['data']
    
    # 数据完整性检查
    total_pixels = ref_data.size
    # 使用np.sum代替np.count_nonzero，减少临时数组创建
    valid_mask = ~np.isnan(ref_data)
    valid_pixels = np.sum(valid_mask)
    invalid_pixels = total_pixels - valid_pixels
    valid_ratio = valid_pixels / total_pixels
    
    results['stats']['total_pixels'] = total_pixels
    results['stats']['valid_pixels'] = valid_pixels
    results['stats']['invalid_pixels'] = invalid_pixels
    results['stats']['valid_ratio'] = valid_ratio
    
    # 噪声数据处理：应用中值滤波减少椒盐噪声影响
    filtered_ref_data = ref_data.copy()
    if ref_data.ndim >= 2:
        # 对每个仰角层独立进行滤波处理
        for sweep_idx in range(radar.nsweeps):
            start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
            end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
            sweep_data = filtered_ref_data[start_ray:end_ray+1, :]
            if sweep_data.shape[0] > 3 and sweep_data.shape[1] > 3:  # 确保有足够数据进行滤波
                # 只对有效数据进行滤波，保持NaN不变
                valid_sweep_mask = ~np.isnan(sweep_data)
                if np.sum(valid_sweep_mask) > 9:  # 至少需要9个有效像素
                    # 创建临时数组，将NaN替换为0进行滤波，然后恢复
                    temp_data = sweep_data.copy()
                    temp_data[~valid_sweep_mask] = 0
                    filtered_temp = median_filter(temp_data, size=3)
                    # 使用np.where避免链式索引赋值导致的read-only错误
                    filtered_ref_data[start_ray:end_ray+1, :] = np.where(valid_sweep_mask, filtered_temp, filtered_ref_data[start_ray:end_ray+1, :])
    
    results['stats']['noise_reduction_applied'] = True
    
    # 缺失值容错：动态调整有效像素比例阈值
    # 如果缺失值集中在局部区域，可能是传感器局部故障
    dynamic_threshold = 0.5
    if valid_ratio < 0.5 and valid_ratio >= 0.3:
        # 检查缺失值的空间分布
        if ref_data.ndim >= 2:
            # 计算每个仰角层的有效像素比例
            layer_valid_ratios = []
            for sweep_idx in range(radar.nsweeps):
                start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
                end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
                sweep_data = ref_data[start_ray:end_ray+1, :]
                sweep_valid = np.sum(~np.isnan(sweep_data))
                sweep_total = sweep_data.size
                layer_valid_ratios.append(sweep_valid / sweep_total if sweep_total > 0 else 0)
            
            # 如果大部分层都有较高的有效像素比例，只有少数层缺失严重，可能是传感器局部故障
            good_layers = sum(1 for ratio in layer_valid_ratios if ratio >= 0.7)
            if good_layers >= len(layer_valid_ratios) * 0.7:
                dynamic_threshold = 0.3  # 降低阈值
                results['status'] = 'warning'
                results['issues'].append(f'反射率数据存在局部缺失，可能是传感器局部故障，有效像素比例为 {valid_ratio:.2%}')
            else:
                results['status'] = 'warning'
                results['issues'].append(f'反射率数据完整性差，有效像素比例仅为 {valid_ratio:.2%}')
    elif valid_ratio < 0.3:
        results['status'] = 'warning'
        results['issues'].append(f'反射率数据完整性严重不足，有效像素比例仅为 {valid_ratio:.2%}')
    
    # 更新有效数据
    valid_mask = ~np.isnan(filtered_ref_data)
    valid_pixels = np.sum(valid_mask)
    
    # 有效性范围检查 (典型范围: -30 dBZ 到 80 dBZ)
    min_valid = -30.0
    max_valid = 80.0
    
    # 只计算一次统计量，使用滤波后的数据
    data_min = np.nanmin(filtered_ref_data)
    data_max = np.nanmax(filtered_ref_data)
    results['stats']['data_range'] = [data_min, data_max]
    
    if data_min < min_valid:
        results['status'] = 'warning'
        results['issues'].append(f'反射率最小值 {data_min:.2f} dBZ 低于有效范围下限 {min_valid} dBZ')
    
    if data_max > max_valid:
        results['status'] = 'warning'
        results['issues'].append(f'反射率最大值 {data_max:.2f} dBZ 高于有效范围上限 {max_valid} dBZ')
    
    # 增强的异常值检测（IQR+标准差+MAD三方法融合）
    try:
        if valid_pixels > 0:
            valid_data = filtered_ref_data[valid_mask]
            
            # IQR方法
            Q1 = np.percentile(valid_data, 25)
            Q3 = np.percentile(valid_data, 75)
            IQR = Q3 - Q1
            iqr_lower = Q1 - 1.5 * IQR
            iqr_upper = Q3 + 1.5 * IQR
            
            # 标准差方法
            mean_ref = np.mean(valid_data)
            std_ref = np.std(valid_data)
            std_lower = mean_ref - 3.0 * std_ref
            std_upper = mean_ref + 3.0 * std_ref
            
            # MAD方法（对异常值更鲁棒）
            median_ref = np.median(valid_data)
            mad = np.median(np.abs(valid_data - median_ref))
            mad_lower = median_ref - 3.5 * mad
            mad_upper = median_ref + 3.5 * mad
            
            # 综合异常值判定：至少两种方法检测到
            outlier_mask_iqr = (valid_data < iqr_lower) | (valid_data > iqr_upper)
            outlier_mask_std = (valid_data < std_lower) | (valid_data > std_upper)
            outlier_mask_mad = (valid_data < mad_lower) | (valid_data > mad_upper)
            outlier_mask = np.sum([outlier_mask_iqr, outlier_mask_std, outlier_mask_mad], axis=0) >= 2
            
            outliers = np.sum(outlier_mask)
            outlier_ratio = outliers / len(valid_data)
            
            results['stats']['outlier_detection'] = {
                'iqr': {'lower': iqr_lower, 'upper': iqr_upper, 'outliers': np.sum(outlier_mask_iqr)},
                'std': {'mean': mean_ref, 'std': std_ref, 'outliers': np.sum(outlier_mask_std)},
                'mad': {'median': median_ref, 'mad': mad, 'outliers': np.sum(outlier_mask_mad)},
                'combined': {'outliers': outliers, 'ratio': outlier_ratio}
            }
            
            if outlier_ratio > 0.1:
                results['status'] = 'warning'
                results['issues'].append(f'反射率异常值比例较高 ({outlier_ratio:.2%})，可能存在数据质量问题或极端天气')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'异常值检测过程中发生异常: {str(e)}')
    
    # 时空一致性检查
    try:
        # 垂直方向一致性检查（相邻仰角层差异）
        if radar.nsweeps > 1:
            vertical_diffs = []
            for i in range(radar.nsweeps - 1):
                # 获取当前扫描层数据
                start_ray1 = radar.sweep_start_ray_index['data'][i]
                end_ray1 = radar.sweep_end_ray_index['data'][i]
                sweep_data1 = filtered_ref_data[start_ray1:end_ray1+1, :]
                
                # 获取下一个扫描层数据
                start_ray2 = radar.sweep_start_ray_index['data'][i+1]
                end_ray2 = radar.sweep_end_ray_index['data'][i+1]
                sweep_data2 = filtered_ref_data[start_ray2:end_ray2+1, :]
                
                # 计算两个扫描层的平均反射率
                mean1 = np.nanmean(sweep_data1)
                mean2 = np.nanmean(sweep_data2)
                
                if not np.isnan(mean1) and not np.isnan(mean2):
                    vertical_diffs.append(abs(mean1 - mean2))
            
            if vertical_diffs:
                mean_vertical_diff = np.mean(vertical_diffs)
                max_vertical_diff = np.max(vertical_diffs)
                
                results['stats']['vertical_consistency'] = {
                    'mean_diff': mean_vertical_diff,
                    'max_diff': max_vertical_diff
                }
                
                if max_vertical_diff > 30.0:
                    results['status'] = 'warning'
                    results['issues'].append(f'检测到反射率垂直方向异常跳变 (最大差异: {max_vertical_diff:.2f} dBZ)')
        
        # 水平方向梯度检查（方位角/距离方向）
        for sweep_idx in range(radar.nsweeps):
            start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
            end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
            sweep_data = filtered_ref_data[start_ray:end_ray+1, :]
            
            if sweep_data.shape[0] > 1 and sweep_data.shape[1] > 1:
                dx = np.abs(np.diff(sweep_data, axis=0))  # 方位角方向梯度
                dy = np.abs(np.diff(sweep_data, axis=1))  # 距离方向梯度
                
                grad_threshold = 20.0  # dBZ/km强梯度阈值
                strong_gradient_pixels = np.sum(dx > grad_threshold) + np.sum(dy > grad_threshold)
                total_gradient_pixels = dx.size + dy.size
                gradient_ratio = strong_gradient_pixels / total_gradient_pixels if total_gradient_pixels > 0 else 0
                
                if sweep_idx == 0:  # 记录第一个仰角层梯度用于综合评估
                    results['stats']['horizontal_gradient'] = {
                        'strong_gradient_ratio': gradient_ratio,
                        'grad_threshold': grad_threshold
                    }
                
                if gradient_ratio > 0.05:
                    results['status'] = 'warning'
                    results['issues'].append(f'检测到强反射率梯度区域，可能存在强风切变，强梯度像素比例为 {gradient_ratio:.2%}')
                    break
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'时空一致性检查异常: {str(e)}')
    
    # 极端天气特征检测
    try:
        # 1. 强反射率区域检测（可能对应冰雹、强降水）
        severe_thresholds = {
            'moderate_precipitation': 40.0,  # 中等降水
            'heavy_precipitation': 50.0,     # 强降水
            'intense_precipitation': 60.0,   # 极强降水
            'hail_possible': 70.0            # 可能有冰雹
        }
        
        severe_stats = {}
        for threshold_name, threshold_value in severe_thresholds.items():
            severe_pixels = np.sum(ref_data > threshold_value)
            severe_ratio = severe_pixels / valid_pixels if valid_pixels > 0 else 0
            severe_stats[threshold_name] = {
                'pixels': severe_pixels,
                'ratio': severe_ratio
            }
            
            if severe_ratio > 0.01:  # 强反射率区域超过1%时发出警告
                results['status'] = 'warning'
                results['issues'].append(f'检测到 {threshold_name.replace("_", " ")} 区域，反射率 > {threshold_value} dBZ 的像素比例为 {severe_ratio:.2%}')
        
        results['stats']['severe_weather'] = severe_stats
        
        # 2. 反射率梯度分析（指示强风切变）
        if ref_data.ndim >= 2:  # 确保数据有足够的维度
            # 对每个仰角层进行梯度分析
            total_strong_gradient = 0
            total_gradient = 0
            
            for sweep_idx in range(radar.nsweeps):
                start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
                end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
                single_sweep = filtered_ref_data[start_ray:end_ray+1, :]  # 使用滤波后的数据
                
                # 计算水平梯度
                if single_sweep.shape[0] > 1 and single_sweep.shape[1] > 1:
                    dx = np.abs(np.diff(single_sweep, axis=0))
                    dy = np.abs(np.diff(single_sweep, axis=1))
                    
                    # 计算梯度超过阈值的区域
                    grad_threshold = 20.0  # dBZ/km，指示强梯度
                    total_strong_gradient += np.sum(dx > grad_threshold) + np.sum(dy > grad_threshold)
                    total_gradient += dx.size + dy.size
            
            gradient_ratio = total_strong_gradient / total_gradient if total_gradient > 0 else 0
            
            results['stats']['strong_gradient_ratio'] = gradient_ratio
            
            if gradient_ratio > 0.05:  # 强梯度区域超过5%时发出警告
                results['status'] = 'warning'
                results['issues'].append(f'检测到强反射率梯度区域，可能存在强风切变，强梯度像素比例为 {gradient_ratio:.2%}')
        
        # 3. 空间分布特征识别 - 升级为多百分位极端天气指数
        if valid_pixels > 0:
            # 计算多百分位数（更全面地评估极端值分布）
            p90 = np.percentile(ref_data[~np.isnan(ref_data)], 90)
            p95 = np.percentile(ref_data[~np.isnan(ref_data)], 95)
            p99 = np.percentile(ref_data[~np.isnan(ref_data)], 99)
            p99_9 = np.percentile(ref_data[~np.isnan(ref_data)], 99.9)
            
            results['stats']['percentiles'] = {
                'p90': p90,
                'p95': p95,
                'p99': p99,
                'p99_9': p99_9
            }
            
            # 升级的极端天气指数（0-7级综合评估）
            extreme_index = 0
            
            # 基于99%百分位数的强降水评估（0-3分）
            if p99 > 70.0:
                extreme_index += 3
            elif p99 > 60.0:
                extreme_index += 2
            elif p99 > 50.0:
                extreme_index += 1
            
            # 基于99.9%百分位数的极端天气评估（0-3分）
            if p99_9 > 80.0:
                extreme_index += 3
            elif p99_9 > 70.0:
                extreme_index += 2
            elif p99_9 > 60.0:
                extreme_index += 1
            
            # 基于95%百分位数的大面积强降水评估（0-1分）
            if p95 > 50.0:
                extreme_index += 1
            
            results['stats']['extreme_weather_index'] = extreme_index
            
            if extreme_index >= 2:
                results['status'] = 'warning'
                results['issues'].append(f'极端天气指数为 {extreme_index}/7，可能存在强对流天气')
            if extreme_index >= 5:
                results['status'] = 'warning'
                results['issues'].append(f'极端天气指数为 {extreme_index}/7，可能存在极端强对流或冰雹天气')
                
    except Exception as e:
        # 如果在极端天气检测过程中发生错误，记录警告但不影响整体检查
        results['status'] = 'warning'
        results['issues'].append(f'极端天气特征检测过程中发生异常: {str(e)}')
    
    return results


def check_radial_velocity(radar):
    """
    检查径向速度的合理区间、异常值检测及极端天气特征
    
    参数:
        radar (Radar): 雷达数据对象
    
    返回:
        dict: 径向速度检查结果，包含状态和详细信息
    """
    results = {
        'parameter': 'radial_velocity',
        'status': 'pass',
        'issues': [],
        'stats': {}
    }
    
    if 'velocity' not in radar.fields and 'radial_velocity' not in radar.fields:
        results['status'] = 'info'
        results['issues'].append('径向速度字段不存在')
        return results
    
    # 确定径向速度字段名称
    vel_field = 'velocity' if 'velocity' in radar.fields else 'radial_velocity'
    
    try:
        # 直接使用原始数据的引用，避免不必要的副本
        vel_data = radar.fields[vel_field]['data']
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'访问径向速度数据失败: {str(e)}')
        return results
    
    # 数据完整性检查
    total_pixels = vel_data.size
    valid_mask = ~np.isnan(vel_data)
    valid_pixels = np.sum(valid_mask)
    valid_ratio = valid_pixels / total_pixels
    
    results['stats']['total_pixels'] = total_pixels
    results['stats']['valid_pixels'] = valid_pixels
    results['stats']['valid_ratio'] = valid_ratio
    
    # 噪声数据处理：应用中值滤波减少椒盐噪声影响
    filtered_vel_data = vel_data.copy()
    if vel_data.ndim >= 2:
        # 对每个仰角层独立进行滤波处理
        for sweep_idx in range(radar.nsweeps):
            start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
            end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
            sweep_data = filtered_vel_data[start_ray:end_ray+1, :]
            if sweep_data.shape[0] > 3 and sweep_data.shape[1] > 3:  # 确保有足够数据进行滤波
                # 只对有效数据进行滤波，保持NaN不变
                valid_sweep_mask = ~np.isnan(sweep_data)
                if np.sum(valid_sweep_mask) > 9:  # 至少需要9个有效像素
                    # 创建临时数组，将NaN替换为0进行滤波，然后恢复
                    temp_data = sweep_data.copy()
                    temp_data[~valid_sweep_mask] = 0
                    filtered_temp = median_filter(temp_data, size=3)
                    # 使用np.where避免链式索引赋值导致的read-only错误
                    filtered_vel_data[start_ray:end_ray+1, :] = np.where(valid_sweep_mask, filtered_temp, filtered_vel_data[start_ray:end_ray+1, :])
    
    # 缺失值容错：动态调整有效像素比例阈值
    # 如果缺失值集中在局部区域，可能是传感器局部故障
    dynamic_threshold = 0.5
    if valid_ratio < 0.5 and valid_ratio >= 0.3:
        # 检查缺失值的空间分布
        if vel_data.ndim >= 2:
            # 计算每个仰角层的有效像素比例
            layer_valid_ratios = []
            for sweep_idx in range(radar.nsweeps):
                start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
                end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
                sweep_data = vel_data[start_ray:end_ray+1, :]
                sweep_valid = np.sum(~np.isnan(sweep_data))
                sweep_total = sweep_data.size
                layer_valid_ratios.append(sweep_valid / sweep_total if sweep_total > 0 else 0)
            
            # 如果大部分层都有较高的有效像素比例，只有少数层缺失严重，可能是传感器局部故障
            good_layers = sum(1 for ratio in layer_valid_ratios if ratio >= 0.7)
            if good_layers >= len(layer_valid_ratios) * 0.7:
                dynamic_threshold = 0.3  # 降低阈值
                results['status'] = 'warning'
                results['issues'].append(f'径向速度数据存在局部缺失，可能是传感器局部故障，有效像素比例为 {valid_ratio:.2%}')
            else:
                results['status'] = 'warning'
                results['issues'].append(f'径向速度数据完整性差，有效像素比例仅为 {valid_ratio:.2%}')
    elif valid_ratio < 0.3:
        results['status'] = 'warning'
        results['issues'].append(f'径向速度数据完整性严重不足，有效像素比例仅为 {valid_ratio:.2%}')
    elif valid_ratio < dynamic_threshold:
        results['status'] = 'warning'
        results['issues'].append(f'径向速度数据完整性差，有效像素比例仅为 {valid_ratio:.2%}')
    
    # 合理区间检查 (典型范围: -60 m/s 到 60 m/s)
    min_valid = -60.0
    max_valid = 60.0
    
    # 更新有效数据掩码
    valid_mask = ~np.isnan(filtered_vel_data)
    valid_pixels = np.sum(valid_mask)
    
    try:
        data_min = np.nanmin(filtered_vel_data)
        data_max = np.nanmax(filtered_vel_data)
        
        results['stats']['data_range'] = [data_min, data_max]
        
        if data_min < min_valid:
            results['status'] = 'warning'
            results['issues'].append(f'径向速度最小值 {data_min:.2f} m/s 低于合理范围下限 {min_valid} m/s')
        
        if data_max > max_valid:
            results['status'] = 'warning'
            results['issues'].append(f'径向速度最大值 {data_max:.2f} m/s 高于合理范围上限 {max_valid} m/s')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'计算径向速度范围失败: {str(e)}')
    
    # 增强的异常值检测（IQR+标准差+MAD三方法融合）
    try:
        # 从滤波后的数据中提取有效数据
        valid_data = filtered_vel_data[valid_mask]
        
        # 如果没有有效数据，跳过异常值检测
        if len(valid_data) == 0:
            return results
        
        # IQR方法
        Q1 = np.percentile(valid_data, 25)
        Q3 = np.percentile(valid_data, 75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR
        
        # 标准差方法
        mean_vel = np.mean(valid_data)
        std_vel = np.std(valid_data)
        std_lower = mean_vel - 3.0 * std_vel
        std_upper = mean_vel + 3.0 * std_vel
        
        # MAD方法（对异常值更鲁棒）
        median_vel = np.median(valid_data)
        mad = np.median(np.abs(valid_data - median_vel))
        mad_lower = median_vel - 3.5 * mad
        mad_upper = median_vel + 3.5 * mad
        
        # 综合三种方法的结果：至少两种方法检测到才判定为异常值
        outlier_mask_iqr = (valid_data < iqr_lower) | (valid_data > iqr_upper)
        outlier_mask_std = (valid_data < std_lower) | (valid_data > std_upper)
        outlier_mask_mad = (valid_data < mad_lower) | (valid_data > mad_upper)
        outlier_mask = np.sum([outlier_mask_iqr, outlier_mask_std, outlier_mask_mad], axis=0) >= 2
        
        outliers = np.sum(outlier_mask)
        outlier_ratio = outliers / len(valid_data)
        
        results['stats']['outlier_detection'] = {
            'iqr': {'lower': iqr_lower, 'upper': iqr_upper, 'outliers': np.sum(outlier_mask_iqr)},
            'std': {'mean': mean_vel, 'std': std_vel, 'outliers': np.sum(outlier_mask_std)},
            'mad': {'median': median_vel, 'mad': mad, 'outliers': np.sum(outlier_mask_mad)},
            'combined': {'outliers': outliers, 'ratio': outlier_ratio}
        }
        
        if outlier_ratio > 0.1:
            results['status'] = 'warning'
            results['issues'].append(f'径向速度异常值比例较高 ({outlier_ratio:.2%})，可能存在数据质量问题或极端天气')
    except Exception as e:
        # 如果在异常值检测过程中发生错误，记录警告
        results['status'] = 'warning'
        results['issues'].append(f'径向速度异常值检测失败: {str(e)}')
    
    # 时空一致性检查
    try:
        # 垂直方向一致性检查（相邻仰角层差异）
        if radar.nsweeps > 1:
            vertical_diffs = []
            for i in range(radar.nsweeps - 1):
                # 获取当前扫描层数据
                start_ray1 = radar.sweep_start_ray_index['data'][i]
                end_ray1 = radar.sweep_end_ray_index['data'][i]
                sweep_data1 = filtered_vel_data[start_ray1:end_ray1+1, :]
                
                # 获取下一个扫描层数据
                start_ray2 = radar.sweep_start_ray_index['data'][i+1]
                end_ray2 = radar.sweep_end_ray_index['data'][i+1]
                sweep_data2 = filtered_vel_data[start_ray2:end_ray2+1, :]
                
                # 计算两个扫描层的平均径向速度
                mean1 = np.nanmean(sweep_data1)
                mean2 = np.nanmean(sweep_data2)
                
                if not np.isnan(mean1) and not np.isnan(mean2):
                    vertical_diffs.append(abs(mean1 - mean2))
            
            if vertical_diffs:
                mean_vertical_diff = np.mean(vertical_diffs)
                max_vertical_diff = np.max(vertical_diffs)
                
                results['stats']['vertical_consistency'] = {
                    'mean_diff': mean_vertical_diff,
                    'max_diff': max_vertical_diff
                }
                
                if max_vertical_diff > 40.0:  # 径向速度垂直差异阈值
                    results['status'] = 'warning'
                    results['issues'].append(f'检测到径向速度垂直方向异常跳变 (最大差异: {max_vertical_diff:.2f} m/s)')
        
        # 水平方向梯度检查（方位角/距离方向）
        for sweep_idx in range(radar.nsweeps):
            start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
            end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
            sweep_data = filtered_vel_data[start_ray:end_ray+1, :]
            
            if sweep_data.shape[0] > 1 and sweep_data.shape[1] > 1:
                dx = np.abs(np.diff(sweep_data, axis=0))  # 方位角方向梯度
                dy = np.abs(np.diff(sweep_data, axis=1))  # 距离方向梯度
                
                grad_threshold = 20.0  # m/s/km强梯度阈值
                strong_gradient_pixels = np.sum(dx > grad_threshold) + np.sum(dy > grad_threshold)
                total_gradient_pixels = dx.size + dy.size
                gradient_ratio = strong_gradient_pixels / total_gradient_pixels if total_gradient_pixels > 0 else 0
                
                if sweep_idx == 0:  # 记录第一个仰角层梯度用于综合评估
                    results['stats']['horizontal_gradient'] = {
                        'strong_gradient_ratio': gradient_ratio,
                        'grad_threshold': grad_threshold
                    }
                
                if gradient_ratio > 0.05:
                    results['status'] = 'warning'
                    results['issues'].append(f'检测到强径向速度梯度区域，可能存在强风切变，强梯度像素比例为 {gradient_ratio:.2%}')
                    break
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'径向速度时空一致性检查异常: {str(e)}')
    
    # 极端天气特征检测
    try:
        if vel_data.ndim >= 2 and valid_pixels > 0:  # 确保数据有足够的维度和有效像素
            # 选择第一个仰角的数据进行分析
            sweep_idx = 0
            start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
            end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
            single_sweep = filtered_vel_data[start_ray:end_ray+1, :]
            
            if single_sweep.shape[0] > 1 and single_sweep.shape[1] > 1:
                # 1. 速度切变检测（指示强风切变或龙卷风）
                dx = np.abs(np.diff(single_sweep, axis=0))
                dy = np.abs(np.diff(single_sweep, axis=1))
                
                # 计算切变超过阈值的区域
                shear_threshold = 30.0  # m/s/km，强切变阈值
                strong_shear_pixels = np.sum(dx > shear_threshold) + np.sum(dy > shear_threshold)
                total_shear_pixels = dx.size + dy.size
                shear_ratio = strong_shear_pixels / total_shear_pixels if total_shear_pixels > 0 else 0
                
                results['stats']['strong_shear_ratio'] = shear_ratio
                
                if shear_ratio > 0.03:  # 强切变区域超过3%时发出警告
                    results['status'] = 'warning'
                    results['issues'].append(f'检测到强速度切变区域，可能存在强风切变或龙卷风，强切变像素比例为 {shear_ratio:.2%}')
                
                # 2. 中气旋特征检测（正负速度对）
                # 计算正负速度的标准差
                pos_vel = valid_data[valid_data > 0]
                neg_vel = valid_data[valid_data < 0]
                
                if len(pos_vel) > 0 and len(neg_vel) > 0:
                    pos_std = np.std(pos_vel)
                    neg_std = np.std(neg_vel)
                    
                    results['stats']['positive_velocity_std'] = pos_std
                    results['stats']['negative_velocity_std'] = neg_std
                    
                    # 检查是否存在强的正负速度对
                    if np.max(pos_vel) > 20.0 and np.abs(np.min(neg_vel)) > 20.0:
                        results['status'] = 'warning'
                        results['issues'].append('检测到强的正负速度对，可能存在中气旋或强对流天气系统')
    except Exception as e:
        # 如果在极端天气检测过程中发生错误，记录警告但不影响整体检查
        results['status'] = 'warning'
        results['issues'].append(f'径向速度极端天气特征检测失败: {str(e)}')
    
    return results


def check_spectral_width(radar):
    """
    检查速度谱宽的数值范围、数据分布及极端天气特征
    
    参数:
        radar (Radar): 雷达数据对象
    
    返回:
        dict: 速度谱宽检查结果，包含状态和详细信息
    """
    results = {
        'parameter': 'spectral_width',
        'status': 'pass',
        'issues': [],
        'stats': {}
    }
    
    if 'spectral_width' not in radar.fields:
        results['status'] = 'info'
        results['issues'].append('速度谱宽字段不存在')
        return results
    
    try:
        sw_data = radar.fields['spectral_width']['data']
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'访问速度谱宽数据失败: {str(e)}')
        return results
    
    # 数据完整性检查
    total_pixels = sw_data.size
    # 使用np.sum代替np.count_nonzero，减少临时数组创建
    valid_mask = ~np.isnan(sw_data)
    valid_pixels = np.sum(valid_mask)
    valid_ratio = valid_pixels / total_pixels
    
    results['stats']['total_pixels'] = total_pixels
    results['stats']['valid_pixels'] = valid_pixels
    results['stats']['valid_ratio'] = valid_ratio
    
    # 噪声数据处理：应用中值滤波减少椒盐噪声影响
    filtered_sw_data = sw_data.copy()
    if sw_data.ndim >= 2:
        # 对每个仰角层独立进行滤波处理
        for sweep_idx in range(radar.nsweeps):
            start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
            end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
            sweep_data = filtered_sw_data[start_ray:end_ray+1, :]
            if sweep_data.shape[0] > 3 and sweep_data.shape[1] > 3:  # 确保有足够数据进行滤波
                # 只对有效数据进行滤波，保持NaN不变
                valid_sweep_mask = ~np.isnan(sweep_data)
                if np.sum(valid_sweep_mask) > 9:  # 至少需要9个有效像素
                    # 创建临时数组，将NaN替换为0进行滤波，然后恢复
                    temp_data = sweep_data.copy()
                    temp_data[~valid_sweep_mask] = 0
                    filtered_temp = median_filter(temp_data, size=3)
                    # 使用np.where避免链式索引赋值导致的read-only错误
                    filtered_sw_data[start_ray:end_ray+1, :] = np.where(valid_sweep_mask, filtered_temp, filtered_sw_data[start_ray:end_ray+1, :])
    
    # 缺失值容错：动态调整有效像素比例阈值
    # 如果缺失值集中在局部区域，可能是传感器局部故障
    dynamic_threshold = 0.5
    if valid_ratio < 0.5 and valid_ratio >= 0.3:
        # 检查缺失值的空间分布
        if sw_data.ndim >= 2:
            # 计算每个仰角层的有效像素比例
            layer_valid_ratios = []
            for sweep_idx in range(radar.nsweeps):
                start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
                end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
                sweep_data = sw_data[start_ray:end_ray+1, :]
                sweep_valid = np.sum(~np.isnan(sweep_data))
                sweep_total = sweep_data.size
                layer_valid_ratios.append(sweep_valid / sweep_total if sweep_total > 0 else 0)
            
            # 如果大部分层都有较高的有效像素比例，只有少数层缺失严重，可能是传感器局部故障
            good_layers = sum(1 for ratio in layer_valid_ratios if ratio >= 0.7)
            if good_layers >= len(layer_valid_ratios) * 0.7:
                dynamic_threshold = 0.3  # 降低阈值
                results['status'] = 'warning'
                results['issues'].append(f'速度谱宽数据存在局部缺失，可能是传感器局部故障，有效像素比例为 {valid_ratio:.2%}')
            else:
                results['status'] = 'warning'
                results['issues'].append(f'速度谱宽数据完整性差，有效像素比例仅为 {valid_ratio:.2%}')
    elif valid_ratio < 0.3:
        results['status'] = 'warning'
        results['issues'].append(f'速度谱宽数据完整性严重不足，有效像素比例仅为 {valid_ratio:.2%}')
    elif valid_ratio < dynamic_threshold:
        results['status'] = 'warning'
        results['issues'].append(f'速度谱宽数据完整性差，有效像素比例仅为 {valid_ratio:.2%}')
    
    # 数值范围验证 (速度谱宽理论上应为非负值，典型范围: 0 m/s 到 20 m/s)
    min_valid = 0.0
    max_valid = 20.0
    
    # 只计算一次统计量
    try:
        data_min = np.nanmin(filtered_sw_data)
        data_max = np.nanmax(filtered_sw_data)
        mean_sw = np.nanmean(filtered_sw_data)
        std_sw = np.nanstd(filtered_sw_data)
        
        results['stats']['data_range'] = [data_min, data_max]
        results['stats']['mean'] = mean_sw
        results['stats']['std'] = std_sw
        
        if data_min < min_valid:
            results['status'] = 'warning'
            results['issues'].append(f'速度谱宽最小值 {data_min:.2f} m/s 为负值，不符合物理意义')
        
        if data_max > max_valid:
            results['status'] = 'warning'
            results['issues'].append(f'速度谱宽最大值 {data_max:.2f} m/s 超出典型范围上限 {max_valid} m/s')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'计算速度谱宽统计量失败: {str(e)}')
        return results
    
    # 时空一致性检查
    try:
        # 垂直方向一致性检查（相邻仰角层差异）
        if radar.nsweeps > 1:
            vertical_diffs = []
            for i in range(radar.nsweeps - 1):
                # 获取当前扫描层数据
                start_ray1 = radar.sweep_start_ray_index['data'][i]
                end_ray1 = radar.sweep_end_ray_index['data'][i]
                sweep_data1 = filtered_sw_data[start_ray1:end_ray1+1, :]
                
                # 获取下一个扫描层数据
                start_ray2 = radar.sweep_start_ray_index['data'][i+1]
                end_ray2 = radar.sweep_end_ray_index['data'][i+1]
                sweep_data2 = filtered_sw_data[start_ray2:end_ray2+1, :]
                
                # 计算两个扫描层的平均速度谱宽
                mean1 = np.nanmean(sweep_data1)
                mean2 = np.nanmean(sweep_data2)
                
                if not np.isnan(mean1) and not np.isnan(mean2):
                    vertical_diffs.append(abs(mean1 - mean2))
            
            if vertical_diffs:
                mean_vertical_diff = np.mean(vertical_diffs)
                max_vertical_diff = np.max(vertical_diffs)
                
                results['stats']['vertical_consistency'] = {
                    'mean_diff': mean_vertical_diff,
                    'max_diff': max_vertical_diff
                }
                
                if max_vertical_diff > 10.0:  # 谱宽垂直差异阈值
                    results['status'] = 'warning'
                    results['issues'].append(f'检测到速度谱宽垂直方向异常跳变 (最大差异: {max_vertical_diff:.2f} m/s)')
        
        # 水平方向梯度检查（方位角/距离方向）
        for sweep_idx in range(radar.nsweeps):
            start_ray = radar.sweep_start_ray_index['data'][sweep_idx]
            end_ray = radar.sweep_end_ray_index['data'][sweep_idx]
            sweep_data = filtered_sw_data[start_ray:end_ray+1, :]
            
            if sweep_data.shape[0] > 1 and sweep_data.shape[1] > 1:
                dx = np.abs(np.diff(sweep_data, axis=0))  # 方位角方向梯度
                dy = np.abs(np.diff(sweep_data, axis=1))  # 距离方向梯度
                
                grad_threshold = 5.0  # m/s/km强梯度阈值
                strong_gradient_pixels = np.sum(dx > grad_threshold) + np.sum(dy > grad_threshold)
                total_gradient_pixels = dx.size + dy.size
                gradient_ratio = strong_gradient_pixels / total_gradient_pixels if total_gradient_pixels > 0 else 0
                
                if sweep_idx == 0:  # 记录第一个仰角层梯度用于综合评估
                    results['stats']['horizontal_gradient'] = {
                        'strong_gradient_ratio': gradient_ratio,
                        'grad_threshold': grad_threshold
                    }
                
                if gradient_ratio > 0.05:
                    results['status'] = 'warning'
                    results['issues'].append(f'检测到强速度谱宽梯度区域，可能存在强风切变，强梯度像素比例为 {gradient_ratio:.2%}')
                    break
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'速度谱宽时空一致性检查异常: {str(e)}')
    
    # 极端天气特征检测
    try:
        if valid_pixels > 0:
            valid_data = filtered_sw_data[valid_mask]
            
            # 1. 高谱宽区域检测（指示湍流和强风暴）
            high_sw_threshold = 10.0  # m/s，高谱宽阈值
            very_high_sw_threshold = 15.0  # m/s，极高谱宽阈值
            
            high_sw_pixels = np.sum(valid_data > high_sw_threshold)
            very_high_sw_pixels = np.sum(valid_data > very_high_sw_threshold)
            
            high_sw_ratio = high_sw_pixels / valid_pixels
            very_high_sw_ratio = very_high_sw_pixels / valid_pixels
            
            results['stats']['high_spectral_width'] = {
                'high_sw_ratio': high_sw_ratio,
                'very_high_sw_ratio': very_high_sw_ratio
            }
            
            if high_sw_ratio > 0.1:
                results['status'] = 'warning'
                results['issues'].append(f'检测到大面积高谱宽区域，可能存在强湍流或风暴，谱宽 > {high_sw_threshold} m/s 的像素比例为 {high_sw_ratio:.2%}')
            
            if very_high_sw_ratio > 0.05:
                results['status'] = 'warning'
                results['issues'].append(f'检测到极高谱宽区域，可能存在极端天气，谱宽 > {very_high_sw_threshold} m/s 的像素比例为 {very_high_sw_ratio:.2%}')
            
            # 2. 谱宽分布分析
            # 计算不同谱宽区间的分布
            bins = [0, 2, 5, 10, 15, 20, np.inf]
            bin_names = ['very_low', 'low', 'moderate', 'high', 'very_high', 'extreme']
            
            hist, _ = np.histogram(valid_data, bins=bins)
            hist_ratio = hist / valid_pixels
            
            sw_distribution = {name: ratio for name, ratio in zip(bin_names, hist_ratio)}
            results['stats']['distribution'] = sw_distribution
            
            # 3. 谱宽梯度检测（指示强风切变）
            if filtered_sw_data.ndim >= 3 and filtered_sw_data.shape[0] > 0:  # 确保数据有足够的维度
                # 选择第一个仰角的数据进行梯度分析
                single_sweep = filtered_sw_data[0]
                
                if single_sweep.shape[0] > 1 and single_sweep.shape[1] > 1:
                    dx = np.abs(np.diff(single_sweep, axis=0))
                    dy = np.abs(np.diff(single_sweep, axis=1))
                    
                    # 计算梯度超过阈值的区域
                    grad_threshold = 5.0  # m/s/km，强梯度阈值
                    strong_grad_pixels = np.sum(dx > grad_threshold) + np.sum(dy > grad_threshold)
                    total_grad_pixels = dx.size + dy.size
                    grad_ratio = strong_grad_pixels / total_grad_pixels if total_grad_pixels > 0 else 0
                    
                    results['stats']['strong_gradient_ratio'] = grad_ratio
                    
                    if grad_ratio > 0.05:
                        results['status'] = 'warning'
                        results['issues'].append(f'检测到强谱宽梯度区域，可能存在强风切变，强梯度像素比例为 {grad_ratio:.2%}')
            
            # 4. 极端天气指数（基于谱宽特征）
            sw_extreme_index = 0
            if mean_sw > 8.0:
                sw_extreme_index += 2
            elif mean_sw > 5.0:
                sw_extreme_index += 1
            
            if very_high_sw_ratio > 0.01:
                sw_extreme_index += 2
            elif high_sw_ratio > 0.05:
                sw_extreme_index += 1
            
            results['stats']['extreme_weather_index'] = sw_extreme_index
            
            if sw_extreme_index >= 3:
                results['status'] = 'warning'
                results['issues'].append(f'谱宽极端天气指数为 {sw_extreme_index}，可能存在强对流或极端天气')
                
    except Exception as e:
        # 如果在极端天气检测过程中发生错误，记录警告但不影响整体检查
        results['status'] = 'warning'
        results['issues'].append(f'速度谱宽极端天气特征检测失败: {str(e)}')
    
    return results


def check_elevations(radar):
    """
    检查所有仰角的连续性、覆盖范围、角度间隔及一致性
    
    参数:
        radar (Radar): 雷达数据对象
    
    返回:
        dict: 仰角检查结果，包含状态和详细信息
    """
    results = {
        'parameter': 'elevations',
        'status': 'pass',
        'issues': [],
        'stats': {}
    }
    
    if not hasattr(radar, 'elevation'):
        results['status'] = 'fail'
        results['issues'].append('仰角数据不存在')
        return results
    
    try:
        elev_data = radar.elevation['data']
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'访问仰角数据失败: {str(e)}')
        return results
    
    # 基本统计信息
    total_angles = len(elev_data)
    
    # 检查数据是否为空
    if total_angles == 0:
        results['status'] = 'fail'
        results['issues'].append('仰角数据为空')
        return results
    
    # 直接使用np.unique的return_counts参数，避免多次排序
    unique_elev, counts = np.unique(elev_data, return_counts=True)
    sorted_elev = unique_elev  # np.unique返回的数组已经是排序好的
    num_unique = len(unique_elev)
    
    results['stats']['total_angles'] = total_angles
    results['stats']['unique_angles'] = num_unique
    results['stats']['angle_range'] = [np.min(elev_data), np.max(elev_data)]
    results['stats']['counts_per_angle'] = dict(zip(unique_elev.tolist(), counts.tolist()))
    
    # 覆盖范围检查 (典型范围: 0 度到 90 度)
    try:
        min_elev = np.min(elev_data)
        max_elev = np.max(elev_data)
        
        if min_elev < 0.0 or max_elev > 90.0:
            results['status'] = 'warning'
            results['issues'].append(f'仰角范围 {min_elev:.2f}° 到 {max_elev:.2f}° 超出典型范围 (0° 到 90°)')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'计算仰角范围失败: {str(e)}')
    
    # 连续性检查
    if num_unique > 1:
        try:
            # 计算仰角间隔
            intervals = np.diff(sorted_elev)
            min_interval = np.min(intervals)
            max_interval = np.max(intervals)
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            results['stats']['intervals'] = {
                'min': min_interval,
                'max': max_interval,
                'avg': avg_interval,
                'std': std_interval,
                'cv': std_interval / avg_interval if avg_interval > 0 else 0
            }
            
            # 检查间隔是否合理 (典型间隔: 0.1° 到 5°)
            if min_interval < 0.05:  # 间隔过小
                results['status'] = 'warning'
                results['issues'].append(f'仰角间隔过小，最小值为 {min_interval:.3f}°，可能存在数据重复或测量误差')
            
            if max_interval > 5.0:  # 间隔过大
                results['status'] = 'warning'
                results['issues'].append(f'仰角间隔过大，最大值为 {max_interval:.2f}°，可能存在仰角缺失')
            
            # 检查间隔一致性（变异系数）
            if avg_interval > 0 and std_interval / avg_interval > 0.5:
                results['status'] = 'warning'
                results['issues'].append(f'仰角间隔一致性差，变异系数为 {std_interval / avg_interval:.2f}')
                
        except Exception as e:
            results['status'] = 'warning'
            results['issues'].append(f'计算仰角间隔失败: {str(e)}')
    
    # 检查仰角数量是否合理
    if num_unique < 2:
        results['status'] = 'warning'
        results['issues'].append(f'仰角数量过少 ({num_unique} 个)，可能影响数据质量')
    elif num_unique > 20:
        results['status'] = 'info'
        results['issues'].append(f'仰角数量较多 ({num_unique} 个)，请确认是否合理')
    
    # 一致性校验：检查每个仰角的扫描数量是否一致
    try:
        if num_unique > 1:
            min_scans = np.min(counts)
            max_scans = np.max(counts)
            
            if max_scans - min_scans > max_scans * 0.1:  # 扫描数量差异超过10%
                results['status'] = 'warning'
                results['issues'].append(f'不同仰角的扫描数量不一致，最小值为 {min_scans}，最大值为 {max_scans}')
                
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'检查仰角扫描数量一致性失败: {str(e)}')
    
    # 仰角顺序检查：确保仰角按递增顺序排列
    try:
        # 检查原始数据是否按仰角分组排列
        if total_angles > num_unique:
            # 计算每个仰角的起始和结束索引
            angle_ranges = []
            current_angle = elev_data[0]
            start_idx = 0
            
            for i in range(1, total_angles):
                if elev_data[i] != current_angle:
                    angle_ranges.append((current_angle, start_idx, i-1))
                    current_angle = elev_data[i]
                    start_idx = i
            
            # 添加最后一个仰角范围
            angle_ranges.append((current_angle, start_idx, total_angles-1))
            
            # 检查仰角顺序是否递增
            angles = [r[0] for r in angle_ranges]
            if not all(angles[i] < angles[i+1] for i in range(len(angles)-1)):
                results['status'] = 'warning'
                results['issues'].append('仰角扫描顺序不是严格递增的，可能影响数据处理')
                
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'检查仰角顺序失败: {str(e)}')
    
    return results


def check_azimuths(radar):
    """
    检查所有方位角的连续性、覆盖范围、角度间隔及一致性
    
    参数:
        radar (Radar): 雷达数据对象
    
    返回:
        dict: 方位角检查结果，包含状态和详细信息
    """
    results = {
        'parameter': 'azimuths',
        'status': 'pass',
        'issues': [],
        'stats': {}
    }
    
    if not hasattr(radar, 'azimuth'):
        results['status'] = 'fail'
        results['issues'].append('方位角数据不存在')
        return results
    
    try:
        azim_data = radar.azimuth['data']
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'访问方位角数据失败: {str(e)}')
        return results
    
    # 基本统计信息
    total_angles = len(azim_data)
    
    # 检查数据是否为空
    if total_angles == 0:
        results['status'] = 'fail'
        results['issues'].append('方位角数据为空')
        return results
    
    # 直接使用np.unique的return_counts参数，避免多次排序
    unique_azim, counts = np.unique(azim_data, return_counts=True)
    sorted_azim = unique_azim  # np.unique返回的数组已经是排序好的
    num_unique = len(unique_azim)
    
    results['stats']['total_angles'] = total_angles
    results['stats']['unique_angles'] = num_unique
    results['stats']['angle_range'] = [np.min(azim_data), np.max(azim_data)]
    results['stats']['counts_per_angle'] = dict(zip(unique_azim.tolist(), counts.tolist()))
    
    # 角度范围检查 (典型范围: 0 度到 360 度)
    try:
        min_azim = np.min(azim_data)
        max_azim = np.max(azim_data)
        
        if min_azim < 0.0 or max_azim > 360.0:
            results['status'] = 'warning'
            results['issues'].append(f'方位角范围 {min_azim:.2f}° 到 {max_azim:.2f}° 超出典型范围 (0° 到 360°)')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'计算方位角范围失败: {str(e)}')
    
    # 完整性检查 (是否覆盖了360度范围)
    try:
        if np.max(azim_data) - np.min(azim_data) < 350.0:
            results['status'] = 'warning'
            results['issues'].append(f'方位角覆盖范围不完整，仅覆盖 {np.max(azim_data) - np.min(azim_data):.2f}° (应接近360°)')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'检查方位角覆盖范围失败: {str(e)}')
    
    # 连续性检查
    if num_unique > 1:
        try:
            # 计算方位角间隔
            intervals = np.diff(sorted_azim)
            min_interval = np.min(intervals)
            max_interval = np.max(intervals)
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            results['stats']['intervals'] = {
                'min': min_interval,
                'max': max_interval,
                'avg': avg_interval,
                'std': std_interval,
                'cv': std_interval / avg_interval if avg_interval > 0 else 0
            }
            
            # 检查间隔是否合理 (典型间隔: 0.5° 到 5°)
            if min_interval < 0.1:  # 间隔过小
                results['status'] = 'warning'
                results['issues'].append(f'方位角间隔过小，最小值为 {min_interval:.3f}°，可能存在数据重复或测量误差')
            
            if max_interval > 10.0:  # 间隔过大
                results['status'] = 'warning'
                results['issues'].append(f'方位角间隔过大，最大值为 {max_interval:.2f}°，可能存在方位角缺失')
            
            # 检查间隔一致性（变异系数）
            if avg_interval > 0 and std_interval / avg_interval > 0.5:
                results['status'] = 'warning'
                results['issues'].append(f'方位角间隔一致性差，变异系数为 {std_interval / avg_interval:.2f}')
                
        except Exception as e:
            results['status'] = 'warning'
            results['issues'].append(f'计算方位角间隔失败: {str(e)}')
    
    # 检查方位角数量是否合理
    try:
        # 计算理论上应该有的方位角数量（基于典型间隔）
        expected_count = 360.0 / 1.0  # 假设1°间隔
        if num_unique < expected_count * 0.9:
            results['status'] = 'warning'
            results['issues'].append(f'方位角数量过少 ({num_unique} 个)，可能存在数据缺失')
        elif num_unique > expected_count * 1.1:
            results['status'] = 'info'
            results['issues'].append(f'方位角数量较多 ({num_unique} 个)，请确认是否合理')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'检查方位角数量合理性失败: {str(e)}')
    
    # 一致性校验：检查每个方位角的扫描数量是否一致
    try:
        if num_unique > 1:
            min_scans = np.min(counts)
            max_scans = np.max(counts)
            
            if max_scans - min_scans > max_scans * 0.1:  # 扫描数量差异超过10%
                results['status'] = 'warning'
                results['issues'].append(f'不同方位角的扫描数量不一致，最小值为 {min_scans}，最大值为 {max_scans}')
                
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'检查方位角扫描数量一致性失败: {str(e)}')
    
    # 方位角顺序检查：确保方位角按递增顺序排列
    try:
        # 检查原始数据是否按方位角连续排列
        if total_angles > num_unique:
            # 计算每个仰角对应的方位角范围
            for i in range(1, total_angles):
                # 允许小的角度跳变（如359°到0°的正常过渡）
                diff = azim_data[i] - azim_data[i-1]
                if diff < -350.0 or diff > 10.0:  # 排除正常的360°环绕过渡
                    results['status'] = 'warning'
                    results['issues'].append(f'检测到方位角异常跳变: 从 {azim_data[i-1]:.2f}° 到 {azim_data[i]:.2f}°')
                    break
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'检查方位角顺序失败: {str(e)}')
    
    return results





def check_radar_position(radar):
    """
    检查雷达位置参数的有效性，包括经纬度坐标范围、高度参数合理性、坐标一致性验证
    
    参数:
        radar (Radar): 雷达数据对象
    
    返回:
        dict: 雷达位置检查结果，包含状态和详细信息
    """
    results = {
        'parameter': 'radar_position',
        'status': 'pass',
        'issues': [],
        'stats': {
            'coordinate_system_check': 'pass',
            'elevation_consistency': 'pass'
        }
    }
    
    try:
        # 1. 经纬度检查 (地球坐标范围)
        if 'longitude' not in radar.metadata:
            results['status'] = 'warning'
            results['issues'].append('雷达经度信息不存在')
        else:
            lon = radar.metadata['longitude']
            results['stats']['longitude'] = lon
            
            # 基本范围检查 (-180° 到 180°)
            if lon < -180.0 or lon > 180.0:
                results['status'] = 'fail'
                results['issues'].append(f'雷达经度 {lon:.4f}° 超出有效范围 (-180° 到 180°)')
            
            # 检查是否为有效数值
            if not isinstance(lon, (int, float)) or np.isnan(lon) or np.isinf(lon):
                results['status'] = 'fail'
                results['issues'].append(f'雷达经度值无效: {lon}')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'经度检查过程中发生异常: {str(e)}')
    
    try:
        # 1. 纬度检查 (地球坐标范围)
        if 'latitude' not in radar.metadata:
            results['status'] = 'warning'
            results['issues'].append('雷达纬度信息不存在')
        else:
            lat = radar.metadata['latitude']
            results['stats']['latitude'] = lat
            
            # 基本范围检查 (-90° 到 90°)
            if lat < -90.0 or lat > 90.0:
                results['status'] = 'fail'
                results['issues'].append(f'雷达纬度 {lat:.4f}° 超出有效范围 (-90° 到 90°)')
            
            # 检查是否为有效数值
            if not isinstance(lat, (int, float)) or np.isnan(lat) or np.isinf(lat):
                results['status'] = 'fail'
                results['issues'].append(f'雷达纬度值无效: {lat}')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'纬度检查过程中发生异常: {str(e)}')
    
    try:
        # 2. 高度检查 (地形一致性)
        if 'altitude' not in radar.metadata:
            results['status'] = 'warning'
            results['issues'].append('雷达高度信息不存在')
        else:
            alt = radar.metadata['altitude']
            results['stats']['altitude'] = alt
            
            # 检查是否为有效数值
            if not isinstance(alt, (int, float)) or np.isnan(alt) or np.isinf(alt):
                results['status'] = 'fail'
                results['issues'].append(f'雷达高度值无效: {alt}')
            else:
                # 高度范围检查 (典型范围: -50 米到 5000 米，考虑海平面以下的雷达)
                if alt < -50.0:
                    results['status'] = 'warning'
                    results['issues'].append(f'雷达高度 {alt:.1f} 米过低，可能低于海平面')
                elif alt > 5000.0:
                    results['status'] = 'warning'
                    results['issues'].append(f'雷达高度 {alt:.1f} 米过高，超出典型安装高度')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'高度检查过程中发生异常: {str(e)}')
    
    try:
        # 3. 坐标系统一致性检查
        # 检查是否存在多个坐标系统定义
        coordinate_systems = []
        
        # 检查metadata中的坐标系统信息
        if 'geospatial_lat_min' in radar.metadata and 'geospatial_lon_min' in radar.metadata:
            coordinate_systems.append('metadata_bounds')
        
        # 检查雷达位置和网格位置的一致性
        if hasattr(radar, 'location') and hasattr(radar.location, 'data'):
            location_data = radar.location['data']
            if len(location_data) >= 2:
                results['stats']['location_data'] = location_data.tolist()
                
                # 比较location数据与metadata中的经纬度
                if 'longitude' in radar.metadata and 'latitude' in radar.metadata:
                    lon_diff = abs(location_data[0] - radar.metadata['longitude'])
                    lat_diff = abs(location_data[1] - radar.metadata['latitude'])
                    
                    if lon_diff > 0.01 or lat_diff > 0.01:  # 超过0.01度的差异
                        results['status'] = 'warning'
                        results['stats']['coordinate_system_check'] = 'warning'
                        results['issues'].append(f'雷达位置数据不一致: metadata与location数据差异超过0.01度')
                    coordinate_systems.append('radar_location')
        
        results['stats']['coordinate_systems_detected'] = coordinate_systems
        
        # 4. 检查雷达扫描范围的物理合理性
        if hasattr(radar, 'range') and hasattr(radar.range, 'data'):
            max_range = np.max(radar.range['data'])
            results['stats']['max_range'] = max_range
            
            # 典型雷达最大探测距离不超过500公里
            if max_range > 500000.0:  # 500公里
                results['status'] = 'warning'
                results['issues'].append(f'雷达最大探测距离 {max_range/1000:.1f} 公里，超出典型范围')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'坐标系统一致性检查过程中发生异常: {str(e)}')
    
    return results


def check_extreme_weather(radar):
    """
    综合检查极端天气模式，整合反射率、径向速度和谱宽信息
    
    参数:
        radar (Radar): 雷达数据对象
    
    返回:
        dict: 极端天气检查结果，包含状态和详细信息
    """
    results = {
        'parameter': 'extreme_weather',
        'status': 'pass',
        'issues': [],
        'stats': {
            'weather_types': [],
            'extreme_indices': {},
            'confidence_level': 'low'
        }
    }
    
    try:
        # 初始化各参数的检查结果
        ref_results = None
        vel_results = None
        sw_results = None
        
        # 检查是否有必要的字段
        has_reflectivity = 'reflectivity' in radar.fields
        has_velocity = 'velocity' in radar.fields or 'radial_velocity' in radar.fields
        has_spectral_width = 'spectral_width' in radar.fields
        
        if not any([has_reflectivity, has_velocity, has_spectral_width]):
            results['status'] = 'warning'
            results['issues'].append('缺少必要的极端天气检测字段')
            return results
        
        # 分别获取各参数的检查结果（如果存在）
        if has_reflectivity:
            ref_results = check_reflectivity(radar)
        
        if has_velocity:
            vel_results = check_radial_velocity(radar)
        
        if has_spectral_width:
            sw_results = check_spectral_width(radar)
        
        # 1. 综合极端天气指数计算
        total_index = 0
        component_indices = {}
        
        # 反射率因子贡献
        if ref_results and 'severe_weather' in ref_results['stats']:
            ref_severe = ref_results['stats']['severe_weather']
            ref_index = 0
            
            # 根据强反射率区域比例计算贡献
            if ref_severe.get('hail_possible', {}).get('ratio', 0) > 0.01:
                ref_index += 5
            elif ref_severe.get('heavy_precipitation', {}).get('ratio', 0) > 0.05:
                ref_index += 3
            elif ref_severe.get('moderate_precipitation', {}).get('ratio', 0) > 0.1:
                ref_index += 1
            
            # 考虑反射率梯度
            if ref_results['stats'].get('strong_gradient_ratio', 0) > 0.05:
                ref_index += 2
            
            component_indices['reflectivity'] = ref_index
            total_index += ref_index
        
        # 径向速度贡献
        if vel_results and 'stats' in vel_results:
            vel_index = 0
            
            # 考虑速度切变
            if vel_results['stats'].get('strong_shear_ratio', 0) > 0.03:
                vel_index += 4
            
            # 考虑正负速度对（中气旋特征）
            if 'positive_velocity_std' in vel_results['stats'] and 'negative_velocity_std' in vel_results['stats']:
                if vel_results['stats']['positive_velocity_std'] > 15.0 and vel_results['stats']['negative_velocity_std'] > 15.0:
                    vel_index += 3
            
            component_indices['radial_velocity'] = vel_index
            total_index += vel_index
        
        # 速度谱宽贡献
        if sw_results and 'stats' in sw_results:
            sw_index = 0
            
            # 考虑高谱宽区域
            if sw_results['stats'].get('high_spectral_width', {}).get('very_high_sw_ratio', 0) > 0.05:
                sw_index += 3
            elif sw_results['stats'].get('high_spectral_width', {}).get('high_sw_ratio', 0) > 0.1:
                sw_index += 2
            
            # 考虑谱宽梯度
            if sw_results['stats'].get('strong_gradient_ratio', 0) > 0.05:
                sw_index += 2
            
            component_indices['spectral_width'] = sw_index
            total_index += sw_index
        
        results['stats']['extreme_indices'] = component_indices
        results['stats']['total_extreme_index'] = total_index
        
        # 2. 极端天气类型识别
        weather_types = []
        
        # 龙卷风潜在识别
        tornado_score = 0
        if ref_results and ref_results['stats'].get('strong_gradient_ratio', 0) > 0.05:
            tornado_score += 1
        if vel_results and vel_results['stats'].get('strong_shear_ratio', 0) > 0.03:
            tornado_score += 2
        if sw_results and sw_results['stats'].get('strong_gradient_ratio', 0) > 0.05:
            tornado_score += 1
        
        if tornado_score >= 3:
            weather_types.append('tornado_potential')
            results['status'] = 'warning'
            results['issues'].append('检测到潜在龙卷风特征')
        
        # 强雷暴识别
        thunderstorm_score = 0
        if ref_results and ref_results['stats'].get('severe_weather', {}).get('heavy_precipitation', {}).get('ratio', 0) > 0.05:
            thunderstorm_score += 2
        if sw_results and sw_results['stats'].get('high_spectral_width', {}).get('high_sw_ratio', 0) > 0.1:
            thunderstorm_score += 2
        if vel_results and vel_results['stats'].get('strong_shear_ratio', 0) > 0.02:
            thunderstorm_score += 1
        
        if thunderstorm_score >= 4:
            weather_types.append('severe_thunderstorm')
            results['status'] = 'warning'
            results['issues'].append('检测到强雷暴特征')
        
        # 冰雹潜在识别
        hail_score = 0
        if ref_results and ref_results['stats'].get('severe_weather', {}).get('hail_possible', {}).get('ratio', 0) > 0.01:
            hail_score += 3
        if sw_results and sw_results['stats'].get('high_spectral_width', {}).get('very_high_sw_ratio', 0) > 0.02:
            hail_score += 2
        
        if hail_score >= 4:
            weather_types.append('hail_potential')
            results['status'] = 'warning'
            results['issues'].append('检测到潜在冰雹特征')
        
        # 暴雨识别
        rain_score = 0
        if ref_results and ref_results['stats'].get('severe_weather', {}).get('heavy_precipitation', {}).get('ratio', 0) > 0.1:
            rain_score += 3
        if ref_results and ref_results['stats'].get('severe_weather', {}).get('moderate_precipitation', {}).get('ratio', 0) > 0.2:
            rain_score += 2
        
        if rain_score >= 4:
            weather_types.append('heavy_rain')
            results['status'] = 'warning'
            results['issues'].append('检测到暴雨特征')
        
        # 强风切变识别
        shear_score = 0
        if ref_results and ref_results['stats'].get('strong_gradient_ratio', 0) > 0.08:
            shear_score += 2
        if vel_results and vel_results['stats'].get('strong_shear_ratio', 0) > 0.05:
            shear_score += 3
        if sw_results and sw_results['stats'].get('strong_gradient_ratio', 0) > 0.08:
            shear_score += 2
        
        if shear_score >= 5:
            weather_types.append('strong_wind_shear')
            results['status'] = 'warning'
            results['issues'].append('检测到强风切变特征')
        
        # 强风识别
        strong_wind_score = 0
        if vel_results:
            # 水平速度梯度（强风指示）
            if vel_results['stats'].get('horizontal_gradient', {}).get('strong_gradient_ratio', 0) > 0.08:
                strong_wind_score += 3
            # 垂直速度差异（强对流风指示）
            if vel_results['stats'].get('vertical_consistency', {}).get('max_diff', 0) > 35.0:
                strong_wind_score += 2
            # 速度极值（极端强风）
            if 'data_range' in vel_results['stats']:
                max_vel = abs(vel_results['stats']['data_range'][1])
                min_vel = abs(vel_results['stats']['data_range'][0])
                if max(max_vel, min_vel) > 45.0:
                    strong_wind_score += 3
        
        if strong_wind_score >= 5:
            weather_types.append('strong_wind')
            results['status'] = 'warning'
            results['issues'].append('检测到强风特征')
        
        # 雷电识别
        lightning_score = 0
        if ref_results:
            # 高反射率特征（雷电伴随强回波）
            if ref_results['stats'].get('percentiles', {}).get('p99', 0) > 55.0:
                lightning_score += 3
            # 极高反射率核心
            if ref_results['stats'].get('severe_weather', {}).get('hail_possible', {}).get('ratio', 0) > 0.005:
                lightning_score += 2
        if sw_results:
            # 高谱宽特征（雷电区域湍流强）
            if sw_results['stats'].get('high_spectral_width', {}).get('very_high_sw_ratio', 0) > 0.03:
                lightning_score += 3
        
        if lightning_score >= 5:
            weather_types.append('lightning')
            results['status'] = 'warning'
            results['issues'].append('检测到雷电特征')
        
        results['stats']['weather_types'] = weather_types
        
        # 3. 置信度评估
        # 基于可用数据的完整性和检测到的特征数量
        data_completeness = sum([has_reflectivity, has_velocity, has_spectral_width]) / 3.0
        feature_count = len(weather_types)
        
        if data_completeness >= 0.67 and feature_count >= 2:
            results['stats']['confidence_level'] = 'high'
        elif data_completeness >= 0.67 or feature_count >= 1:
            results['stats']['confidence_level'] = 'medium'
        
        # 4. 生成综合评估
        if total_index >= 10:
            results['status'] = 'warning'
            results['issues'].append(f'极端天气综合指数为 {total_index}，存在高度极端天气风险')
        elif total_index >= 5:
            results['status'] = 'warning'
            results['issues'].append(f'极端天气综合指数为 {total_index}，存在中度极端天气风险')
        
        # 如果没有检测到极端天气但数据完整，更新状态为正常
        if not weather_types and data_completeness >= 0.67:
            results['status'] = 'pass'
        
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'极端天气检查过程中发生异常: {str(e)}')
    
    return results


def _run_check_wrapper(check_tuple):
    """
    辅助函数包装器，用于在多进程中运行单个检查函数
    
    参数:
        check_tuple (tuple): 包含检查函数和雷达数据对象的元组
    
    返回:
        dict: 检查结果
    """
    check_func, radar = check_tuple
    try:
        return check_func(radar)
    except Exception as e:
        # 记录检查过程中的异常
        return {
            'parameter': check_func.__name__.replace('check_', ''),
            'status': 'error',
            'issues': [f'检查过程中发生异常: {str(e)}'],
            'stats': {}
        }


def check_time_consistency(radar):
    """
    检查雷达数据的时间一致性
    
    参数:
        radar (Radar): 雷达数据对象
    
    返回:
        dict: 时间一致性检查结果，包含状态、问题和统计信息
    """
    results = {
        'status': 'pass',
        'issues': [],
        'stats': {}
    }
    
    try:
        # 时间一致性检查（多时间戳数据）
        if hasattr(radar, 'time') and 'data' in radar.time and len(radar.time['data']) > 1:
            time_diffs = np.diff(radar.time['data'])
            mean_time_diff = np.mean(time_diffs)
            std_time_diff = np.std(time_diffs)
            
            results['stats']['time_consistency'] = {
                'mean_time_diff': mean_time_diff,
                'std_time_diff': std_time_diff
            }
            
            if std_time_diff > mean_time_diff * 0.5:
                results['status'] = 'warning'
                results['issues'].append(f'时间戳间隔不一致，可能存在数据采集异常')
    except Exception as e:
        results['status'] = 'warning'
        results['issues'].append(f'时间一致性检查异常: {str(e)}')
    
    return results

@timing_decorator
def quality_check_radar_data(radar, pool=None, use_parallel=True):
    """
    全面检查雷达数据质量，验证所有关键参数
    
    参数:
        radar (Radar): 雷达数据对象
        pool (Pool, optional): 可选的已存在进程池，用于复用进程
        use_parallel (bool): 是否使用并行处理，默认为True
    
    返回:
        dict: 包含所有检查结果的数据质量报告
    """
    print("=== 开始全面雷达数据质量检查 ===")
    
    # 执行全局时间一致性检查（只需要执行一次）
    time_consistency_result = check_time_consistency(radar)
    
    # 执行所有参数检查，按计算复杂度排序（最耗时的放在前面）
    checks = [
        check_reflectivity,  # 最耗时：包含异常值检测、时空一致性检查、极端天气检测
        check_radial_velocity,  # 较耗时：包含异常值检测和时空一致性检查
        check_spectral_width,  # 较耗时：包含异常值检测和时空一致性检查
        check_extreme_weather,  # 中等耗时：综合分析所有参数
        check_azimuths,  # 低耗时：几何参数检查
        check_elevations,  # 低耗时：几何参数检查
        check_radar_position  # 低耗时：位置参数检查
    ]
    
    # 计算复杂度权重，用于负载均衡
    check_weights = {
        check_reflectivity: 5,
        check_radial_velocity: 4,
        check_spectral_width: 4,
        check_extreme_weather: 3,
        check_azimuths: 1,
        check_elevations: 1,
        check_radar_position: 1
    }
    
    # 使用并行处理提高效率
    try:
        if use_parallel:
            # 准备参数元组列表
            check_tuples = [(check_func, radar) for check_func in checks]
            
            # 如果提供了进程池，则复用它；否则使用全局进程池
            if pool is not None:
                all_results = list(pool.imap_unordered(_run_check_wrapper, check_tuples, chunksize=1))
            else:
                # 使用全局进程池，避免重复创建
                global_pool = get_global_pool()
                all_results = list(global_pool.imap_unordered(_run_check_wrapper, check_tuples, chunksize=1))
        else:
            # 直接使用串行处理
            all_results = []
            for check_func in checks:
                try:
                    result = check_func(radar)
                    all_results.append(result)
                except Exception as e:
                    # 记录检查过程中的异常
                    error_result = {
                        'parameter': check_func.__name__.replace('check_', ''),
                        'status': 'error',
                        'issues': [f'检查过程中发生异常: {str(e)}'],
                        'stats': {}
                    }
                    all_results.append(error_result)
    except Exception as e:
        # 并行处理失败时回退到串行处理
        print(f"并行处理失败，回退到串行处理: {e}")
        all_results = []
        for check_func in checks:
            try:
                result = check_func(radar)
                all_results.append(result)
            except Exception as e:
                # 记录检查过程中的异常
                error_result = {
                    'parameter': check_func.__name__.replace('check_', ''),
                    'status': 'error',
                    'issues': [f'检查过程中发生异常: {str(e)}'],
                    'stats': {}
                }
                all_results.append(error_result)
    
    # 将时间一致性检查结果添加到所有检查结果中
    time_consistency_result['parameter'] = 'time_consistency'
    all_results.append(time_consistency_result)
    
    # 生成综合报告
    overall_status = 'pass'
    for result in all_results:
        if result['status'] == 'fail' or result['status'] == 'error':
            overall_status = 'fail'
            break
        elif result['status'] == 'warning' and overall_status != 'fail':
            overall_status = 'warning'
    
    report = {
        'overall_status': overall_status,
        'checks': all_results,
        'timestamp': radar.time['units'] if 'time' in radar.metadata else 'unknown'
    }
    
    # 打印报告摘要
    print(f"\n=== 数据质量检查报告摘要 ===")
    print(f"总体状态: {overall_status}")
    print(f"检查参数数量: {len(all_results)}")
    
    # 打印各参数状态
    print(f"\n各参数检查状态:")
    for result in all_results:
        status_emoji = '✅' if result['status'] == 'pass' else '⚠️' if result['status'] == 'warning' else '❌'
        print(f"  {status_emoji} {result['parameter']}: {result['status']}")
        if result['issues']:
            for issue in result['issues']:
                print(f"     - {issue}")
    
    print(f"\n=== 数据质量检查完成 ===")
    
    return report


@timing_decorator
def load_radar_data(file_path):
    """
    加载雷达数据文件，支持多种格式
    
    参数:
        file_path (str): 雷达数据文件路径
    
    返回:
        Radar or None: 雷达数据对象，如果文件读取失败则返回None
    """
    try:
        # 使用延迟加载的pyart库读取雷达数据
        pyart = get_pyart()
        print(f"尝试使用pyart读取雷达数据文件: {file_path}")
        radar = pyart.io.read(file_path)
        print(f"成功使用pyart读取雷达数据文件: {file_path}")
        return radar
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return None
    except Exception as e:
        print(f"错误: 读取雷达数据文件时发生异常: {e}")
        return None








@timing_decorator
def create_multiple_cappi(file_path, heights, radar=None):
    """
    从 NEXRAD 雷达数据文件创建多个不同高度的 CAPPI 图像
    
    参数:
        file_path (str): NEXRAD 雷达数据文件路径
        heights (list): CAPPI 高度列表，单位为米
        radar (Radar, optional): 已读取的雷达数据对象，用于避免重复读取
    
    返回:
        None: 显示多个高度的 CAPPI 对比图像
    """
    # 参数验证
    if not isinstance(heights, list) or len(heights) == 0:
        print("错误: heights 必须是一个非空列表")
        return
    
    # 批量验证heights中的值是否为正数，避免多次循环
    if not all(isinstance(height, (int, float)) and height > 0 for height in heights):
        print("错误: 所有高度值必须是正数")
        return
    
    # 读取雷达数据（如果未提供）
    if radar is None:
        try:
            pyart = get_pyart()
            radar = pyart.io.read_nexrad_archive(file_path)
        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 不存在")
            return
        except Exception as e:
            print(f"错误: 读取雷达数据文件时发生异常: {e}")
            return
    
    # 计算子图数量
    num_heights = len(heights)
    
    # 创建画布，1行num_heights列，显示所有高度的CAPPI
    fig, axes = plt.subplots(1, num_heights, figsize=(5 * num_heights, 5))
    
    # 确保axes是数组
    if num_heights == 1:
        axes = [axes]
    
    # 使用 grid_from_radars 函数创建 3D 网格
    print("\n=== 使用 grid_from_radars 函数创建 3D 网格 ===")
    
    # 设置网格参数 - 注意 grid_shape 顺序为 (nz, ny, nx)
    nz = 100  # 固定高度层数，确保足够的垂直分辨率
    ny = 200  # y 方向网格数
    nx = 200  # x 方向网格数
    grid_shape = (nz, ny, nx)
    
    # grid_limits 顺序为 (z_limits, y_limits, x_limits)
    grid_limits = (
        (0.0, 6000.0),          # z 方向范围 (米) - 高度层
        (-230000.0, 230000.0),  # y 方向范围 (米) - 南北方向
        (-230000.0, 230000.0),  # x 方向范围 (米) - 东西方向
    )
    
    print(f"创建 3D 网格...")
    print(f"网格形状: {grid_shape} (nz, ny, nx)")
    print(f"网格范围: z{grid_limits[0]}, y{grid_limits[1]}, x{grid_limits[2]}")
    
    try:
        # 使用延迟加载的pyart库创建3D网格
        pyart = get_pyart()
        grid = pyart.map.grid_from_radars(
            (radar,),
            grid_shape=grid_shape,
            grid_limits=grid_limits,
            fields=["reflectivity"],
            refl_field="reflectivity",
            max_refl=80.0
        )
    except Exception as e:
        print(f"错误: 创建3D网格时发生异常: {e}")
        plt.close(fig)  # 关闭未使用的图形
        return
    
    # 获取网格的高度层值
    z_levels = grid.z['data']
    print(f"网格实际高度层: {z_levels} 米")
    
    # 为每个高度创建并绘制 CAPPI
    ref_data_list = []
    
    for i, target_height in enumerate(heights):
        print(f"\n处理 {target_height} 米高度的 CAPPI...")
        
        # 找到网格中最接近目标高度的高度层
        z_index = np.argmin(np.abs(z_levels - target_height))
        actual_height = z_levels[z_index]
        
        print(f"   - 使用网格高度层 {z_index}: {actual_height:.2f} 米 (目标: {target_height} 米)")
        print(f"   - 高度误差: {abs(actual_height - target_height):.2f} 米")
        
        try:
            # 提取该高度层的反射率数据 - 顺序为 (z, y, x)
            ref_data = grid.fields['reflectivity']['data'][z_index, :, :]
            ref_data_list.append(ref_data)
            
            # 打印更详细的反射率统计信息
            print(f"   - 反射率数据形状: {ref_data.shape}")
            print(f"   - 反射率数据范围: {np.nanmin(ref_data):.2f} 至 {np.nanmax(ref_data):.2f} dBZ")
            print(f"   - 反射率数据平均值: {np.nanmean(ref_data):.2f} dBZ")
            print(f"   - 反射率数据标准差: {np.nanstd(ref_data):.2f} dBZ")
            
            # 检查数据是否全为相同值
            unique_values = np.unique(ref_data[~np.isnan(ref_data)])
            print(f"   - 反射率唯一值数量: {len(unique_values)}")
            
            # 使用 matplotlib 直接绘制网格数据
            im = axes[i].imshow(
                ref_data,
                cmap="NWSRef",
                vmin=-5,
                vmax=75,
                origin='lower'
            )
            axes[i].set_title(f"CAPPI at {actual_height:.0f}m")
            axes[i].set_xlabel("距离 (km)")
            axes[i].set_ylabel("方位角")
            
            # 添加颜色条
            plt.colorbar(im, ax=axes[i], label="反射率 (dBZ)")
        except KeyError as e:
            print(f"   - 错误: 缺少反射率数据字段: {e}")
        except Exception as e:
            print(f"   - 错误: 处理反射率数据时发生异常: {e}")
    
    # 对比不同高度CAPPI数据的差异
    print(f"\n=== 不同高度 CAPPI 数据对比 ===")
    for i in range(len(heights)-1):
        for j in range(i+1, len(heights)):
            try:
                # 计算两个高度CAPPI数据的差异
                diff = np.nanmax(np.abs(ref_data_list[i] - ref_data_list[j]))
                print(f"   - {heights[i]}m 与 {heights[j]}m CAPPI 数据最大差异: {diff:.2f} dBZ")
                
                # 计算相关系数
                # 去除NaN值
                ref1 = ref_data_list[i]
                ref2 = ref_data_list[j]
                
                # 创建掩码，只考虑两个数组中都不是NaN的位置
                mask = ~np.isnan(ref1) & ~np.isnan(ref2)
                
                if np.any(mask):
                    # 只使用两个数组中都有效的数据点
                    ref1_valid = ref1[mask]
                    ref2_valid = ref2[mask]
                    
                    # 计算相关系数
                    corr = np.corrcoef(ref1_valid, ref2_valid)[0, 1]
                    print(f"   - {heights[i]}m 与 {heights[j]}m CAPPI 数据相关系数: {corr:.4f}")
                else:
                    print(f"   - {heights[i]}m 与 {heights[j]}m CAPPI 数据无共同有效数据点")
            except Exception as e:
                print(f"   - 错误: 对比 {heights[i]}m 与 {heights[j]}m 数据时发生异常: {e}")
    
    # 调整布局，防止重叠
    plt.tight_layout()
    
    # 显示所有图像
    plt.show()


if __name__ == "__main__":
    # 文件路径设置 - 用户可以根据需要修改
    PATH = 'KATX20130717_195021_V06'  # NEXRAD格式示例文件（当前目录）
    
    try:
        # 加载雷达数据
        radar = load_radar_data(PATH)
        
        if radar is not None:
            # 执行全面数据质量检查
            quality_report = quality_check_radar_data(radar)
            
            # 根据数据质量检查结果决定是否继续处理
            if quality_report['overall_status'] == 'fail':
                print("\n错误: 数据质量检查失败，无法生成 CAPPI 图像")
            else:
                # 生成多个不同高度的CAPPI
                print("\n=== 生成多个高度的CAPPI ===")
                test_heights = [1000, 2000, 3000, 4000, 5000]  # 单位：米
                create_multiple_cappi(PATH, test_heights, radar)
        else:
            print("\n错误: 无法生成 CAPPI 图像，雷达数据读取失败")
    finally:
        # 确保程序结束时关闭全局进程池，避免资源泄漏
        close_global_pool()