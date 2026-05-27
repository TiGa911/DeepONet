import numpy as np

def calculate_coulomb_log(n_e, T, collision_type, is_ion_ion=False):
    """
    计算库仑对数 (lnΛ)
    Args:
        n_e: 电子密度 (m^-3)
        T: 温度 (keV)
        collision_type: 'ee' (电子-电子), 'ei' (电子-离子), 'ii' (离子-离子)
        is_ion_ion: 是否为离子-离子碰撞（仅当collision_type='ii'时有效）
    Returns:
        ln_lambda: 库仑对数
    """
    n_e_normalized = n_e / 1e20  # 将n_e归一化为10^20 m^-3
    
    if collision_type == 'ee':
        ln_lambda = 14.9 - 0.5 * np.log(n_e_normalized) + np.log(T)
    elif collision_type == 'ei':
        ln_lambda = 15.2 - 0.5 * np.log(n_e_normalized) + np.log(T)
    elif collision_type == 'ii':
        if is_ion_ion:
            ln_lambda = 17.3 - 0.5 * np.log(n_e_normalized) + 1.5 * np.log(T)
        else:
            raise ValueError("离子-离子碰撞需设置 is_ion_ion=True")
    else:
        raise ValueError("碰撞类型必须是 'ee', 'ei' 或 'ii'")
    
    return ln_lambda

def parallel_conductivity(n_e, T, collision_type='ei', is_ion_ion=False):
    """
    计算平行电导率 (σ_parallel)
    Args:
        n_e: 电子密度 (m^-3)
        T: 温度 (keV)
        collision_type: 'ee', 'ei', 'ii'
        is_ion_ion: 是否离子-离子碰撞
    Returns:
        sigma_parallel: 平行电导率 (S/m)
    """
    ln_lambda = calculate_coulomb_log(n_e, T, collision_type, is_ion_ion)
    sigma = (T ** 1.5) / (1.65e-9 * ln_lambda)
    return sigma

#===============================
# 示例使用
#===============================
if __name__ == "__main__":
    # 示例参数
    n_e = 5e19    # 电子密度 (m^-3)
    T_e = 2.0     # 电子温度 (keV)
    
    # 计算电子-电子碰撞的平行电导率
    sigma_ee = parallel_conductivity(n_e, T_e, collision_type='ee')
    print(f"电子-电子碰撞平行电导率: {sigma_ee:.2e} S/m")
    
    # 计算电子-离子碰撞的平行电导率
    sigma_ei = parallel_conductivity(n_e, T_e, collision_type='ei')
    print(f"电子-离子碰撞平行电导率: {sigma_ei:.2e} S/m")
    
    # 计算离子-离子碰撞的平行电导率（假设为单电荷离子）
    sigma_ii = parallel_conductivity(n_e, T_e, collision_type='ii', is_ion_ion=True)
    print(f"离子-离子碰撞平行电导率: {sigma_ii:.2e} S/m")