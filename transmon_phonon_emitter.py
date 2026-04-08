"""
Transmon Qubit 行波声子发射仿真
===============================
基于 QuTiP 的量子主方程求解器

物理模型：
--------
- Transmon qubit 通过 IDT 耦合到 1D 行波声子波导
- 使用 Lindblad 主方程描述开放系统动力学
- 高斯微波脉冲实现可控声子发射
- 支持 Time-bin 编码（双脉冲时序）

关键特征（无 JC 模型）：
----------------------
1. 无声子谐振子算符，声子发射通过 qubit 耗散直接描述
2. 行波声子视为不可逆的连续谱环境（Born-Markov 近似）
3. 发射的声子数为通过计算耗散流获得

作者: Kimi Code
日期: 2026-04-08
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免显示窗口阻塞
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import qutip

# 导入物理参数
from parameters import (
    OMEGA_Q, ALPHA, GAMMA_1, GAMMA_PHI, KAPPA_PH,
    OMEGA_D, OMEGA_0, SIGMA, PULSE_CENTER_1, PULSE_CENTER_2,
    T_LIST, T_MAX, ETA_MAX,
    COLOR_POPULATION, COLOR_EFFICIENCY, COLOR_CUMULATIVE, COLOR_PULSE, DPI
)


# ============================================================================
# 算符定义
# ============================================================================

def create_operators(N_levels=3):
    """
    创建 qubit 算符
    
    使用 3 能级近似描述 Transmon（保留 |g⟩, |e⟩, |f⟩）
    
    Parameters
    ----------
    N_levels : int
        考虑的能级数（默认 3：g, e, f）
    
    Returns
    -------
    dict : 包含所有相关算符的字典
    """
    # 降算符 σ_- = |g⟩⟨e| + √2|e⟩⟨f| + ... 
    # 在 Transmon 中，由于非谐性，能级间距不均匀
    sigma_minus = qutip.destroy(N_levels)
    
    # 升算符 σ_+ = σ_-^†
    sigma_plus = sigma_minus.dag()
    
    # 粒子数算符 n = σ_+ σ_-
    n_op = sigma_plus * sigma_minus
    
    # Pauli-z 算符 σ_z = 2n - 1（在 qubit 子空间中）
    # 注意：这里使用投影到 |e⟩⟨e| 的方式定义
    sigma_z = 2 * n_op - qutip.qeye(N_levels)
    
    # 在旋转框架中使用的驱动算符
    # 通常用 σ_x = σ_+ + σ_-
    sigma_x = sigma_plus + sigma_minus
    
    return {
        'sigma_minus': sigma_minus,
        'sigma_plus': sigma_plus,
        'n_op': n_op,
        'sigma_z': sigma_z,
        'sigma_x': sigma_x
    }


# ============================================================================
# Hamiltonian 构建
# ============================================================================

def transmon_hamiltonian(omega_q, alpha, ops):
    """
    构建 Transmon 哈密顿量（旋转框架）
    
    H_0 = ℏ(ω_q - ω_d) n + ℏ(α/2) n(n-1)
    
    在旋转框架中（ω_d = ω_q）：
    H_0 = ℏ(α/2) n(n-1)
    
    Parameters
    ----------
    omega_q : float
        Qubit 频率 (GHz)
    alpha : float
        非谐性 (GHz)，应为负值
    ops : dict
        算符字典
    
    Returns
    -------
    Qobj : 哈密顿量算符
    """
    n = ops['n_op']
    # 非谐性能级修正：E_n = n*ω_q + (α/2)*n*(n-1)
    # 在旋转框架中，第一项被消除，保留非谐性
    H0 = (alpha / 2.0) * n * (n - 1)
    return H0


def gaussian_pulse(t, omega_0, sigma, t_center):
    """
    高斯脉冲包络
    
    Ω(t) = Ω_0 * exp(-(t-t_0)^2 / (2*σ^2))
    
    Parameters
    ----------
    t : float
        时间 (ns)
    omega_0 : float
        脉冲峰值幅度 (GHz)
    sigma : float
        脉冲宽度 (ns)
    t_center : float
        脉冲中心位置 (ns)
    
    Returns
    -------
    float : 脉冲幅度
    """
    return omega_0 * np.exp(-0.5 * ((t - t_center) / sigma) ** 2)


def double_gaussian_pulse(t, omega_0, sigma, t1, t2, separation):
    """
    双高斯脉冲（Time-bin 编码）
    
    Parameters
    ----------
    t : float
        时间 (ns)
    omega_0 : float
        脉冲幅度 (GHz)
    sigma : float
        脉冲宽度 (ns)
    t1 : float
        第一个脉冲中心 (ns)
    t2 : float
        第二个脉冲中心 (ns)
    separation : float
        脉冲间隔 (ns)，用于验证
    
    Returns
    -------
    float : 脉冲幅度
    """
    pulse1 = gaussian_pulse(t, omega_0, sigma, t1)
    pulse2 = gaussian_pulse(t, omega_0, sigma, t2)
    return pulse1 + pulse2


def sech_pulse(t, omega_0, sigma, t_center):
    """
    Sech（双曲正割）脉冲包络
    
    Ω(t) = Ω_0 * sech((t-t_0)/σ) = Ω_0 / cosh((t-t_0)/σ)
    
    Sech 脉冲特性：
    - 时域：比高斯衰减更慢（有长尾）
    - 频域：Lorentzian 线型（比高斯有更宽的低频分量）
    - 在光孤子和某些声子波导中更接近实际模式
    
    Parameters
    ----------
    t : float
        时间 (ns)
    omega_0 : float
        脉冲峰值幅度 (GHz)
    sigma : float
        特征宽度 (ns)，对应 sech 的半宽参数
    t_center : float
        脉冲中心位置 (ns)
    
    Returns
    -------
    float : 脉冲幅度
    """
    from numpy import cosh
    return omega_0 / cosh((t - t_center) / sigma)


def double_sech_pulse(t, omega_0, sigma, t1, t2, separation):
    """
    双 sech 脉冲（Time-bin 编码）
    
    Parameters
    ----------
    t : float
        时间 (ns)
    omega_0 : float
        脉冲幅度 (GHz)
    sigma : float
        脉冲宽度 (ns)
    t1 : float
        第一个脉冲中心 (ns)
    t2 : float
        第二个脉冲中心 (ns)
    separation : float
        脉冲间隔 (ns)
    
    Returns
    -------
    float : 脉冲幅度
    """
    pulse1 = sech_pulse(t, omega_0, sigma, t1)
    pulse2 = sech_pulse(t, omega_0, sigma, t2)
    return pulse1 + pulse2


def drive_hamiltonian(t, args):
    """
    时变驱动哈密顿量 H_d(t) = Ω(t) * σ_x / 2
    
    支持高斯或 sech 脉冲包络
    
    Parameters
    ----------
    t : float
        时间
    args : dict
        包含 sigma_x 算符和脉冲参数
    
    Returns
    -------
    Qobj : 驱动哈密顿量
    """
    sigma_x = args['sigma_x']
    omega_0 = args['omega_0']
    sigma = args['sigma']
    t1 = args['t1']
    t2 = args['t2']
    use_double = args.get('use_double_pulse', False)
    pulse_type = args.get('pulse_type', 'gaussian')
    
    if pulse_type == 'sech':
        if use_double:
            omega_t = double_sech_pulse(t, omega_0, sigma, t1, t2, args.get('separation', 0))
        else:
            omega_t = sech_pulse(t, omega_0, sigma, t1)
    else:  # gaussian
        if use_double:
            omega_t = double_gaussian_pulse(t, omega_0, sigma, t1, t2, args.get('separation', 0))
        else:
            omega_t = gaussian_pulse(t, omega_0, sigma, t1)
    
    # 旋转波近似下，驱动项为 Ω(t) * σ_x / 2
    return 0.5 * omega_t * sigma_x


# ============================================================================
# Lindblad 耗散项
# ============================================================================

def create_lindblad_ops(ops, gamma1, gamma_phi, kappa_ph):
    """
    创建所有 Lindblad 耗散算符
    
    三个独立的耗散通道：
    1. γ1 : qubit 固有弛豫（非辐射损耗）
    2. γφ : 纯退相（无能量损耗的相位信息丢失）
    3. κ_ph : 向行波声子的有用发射
    
    Parameters
    ----------
    ops : dict
        算符字典
    gamma1 : float
        固有弛豫率 (GHz)
    gamma_phi : float
        纯退相率 (GHz)
    kappa_ph : float
        声子发射率 (GHz)
    
    Returns
    -------
    list : [(L1, rate1), (L2, rate2), ...] 形式的列表
    """
    sigma_minus = ops['sigma_minus']
    sigma_z = ops['sigma_z']
    
    # 1. 固有弛豫：L = σ_-, rate = γ1
    # 描述 qubit 到非声子环境的能量损耗（如介质损耗）
    c_ops_gamma1 = np.sqrt(gamma1) * sigma_minus
    
    # 2. 纯退相：L = σ_z, rate = γφ/2
    # 注意：标准 Lindblad 形式中使用 √(γφ/2) * σ_z
    # 这样产生的退相干率正好是 γφ
    c_ops_phi = np.sqrt(gamma_phi / 2) * sigma_z
    
    # 3. 声子发射：L = σ_-, rate = κ_ph
    # 这是向行波声子波导的有用发射
    # 物理图像：qubit 激发 → 通过 IDT 转换为行波声子
    c_ops_phonon = np.sqrt(kappa_ph) * sigma_minus
    
    return [c_ops_gamma1, c_ops_phi, c_ops_phonon]


def compute_phonon_emission_rate(states, kappa_ph, ops):
    """
    计算瞬时声子发射率
    
    Γ_ph(t) = κ_ph * ⟨σ_+(t) σ_-(t)⟩ = κ_ph * P_e(t)
    
    Parameters
    ----------
    states : list
        密度矩阵时间序列（mesolve 返回的 result.states）
    kappa_ph : float
        声子发射率
    ops : dict
        算符字典
    
    Returns
    -------
    array : 声子发射率时间序列 (GHz)
    """
    n_op = ops['n_op']
    # 计算每个时刻的激发态布居
    P_e_array = np.array([qutip.expect(n_op, rho) for rho in states])
    return kappa_ph * P_e_array


# ============================================================================
# 主求解器
# ============================================================================

def simulate_phonon_emission(
    omega_q=OMEGA_Q,
    alpha=ALPHA,
    gamma1=GAMMA_1,
    gamma_phi=GAMMA_PHI,
    kappa_ph=KAPPA_PH,
    omega_0=OMEGA_0,
    sigma=SIGMA,
    t_center=PULSE_CENTER_1,
    t_list=T_LIST,
    use_double_pulse=False,
    t_center_2=PULSE_CENTER_2,
    pulse_separation=None,
    pulse_type='gaussian',
    N_levels=3
):
    """
    执行声子发射仿真
    
    Parameters
    ----------
    ... (见参数定义)
    use_double_pulse : bool
        是否使用双脉冲（Time-bin 编码）
    t_center_2 : float
        第二个脉冲中心（仅在 use_double_pulse=True 时使用）
    pulse_type : str
        脉冲类型: 'gaussian' 或 'sech'
    
    Returns
    -------
    dict : 包含仿真结果的字典
    """
    print("\n" + "=" * 60)
    print("开始声子发射仿真")
    print("=" * 60)
    
    # 创建算符
    ops = create_operators(N_levels)
    print(f"能级数: {N_levels} (g, e, f)")
    print(f"脉冲类型: {pulse_type}")
    
    # 构建哈密顿量
    H0 = transmon_hamiltonian(omega_q, alpha, ops)
    
    # 驱动参数
    drive_args = {
        'sigma_x': ops['sigma_x'],
        'omega_0': omega_0,
        'sigma': sigma,
        't1': t_center,
        't2': t_center_2,
        'use_double_pulse': use_double_pulse,
        'separation': pulse_separation if pulse_separation else abs(t_center_2 - t_center),
        'pulse_type': pulse_type
    }
    
    # 选择脉冲函数
    if pulse_type == 'sech':
        single_pulse_func = sech_pulse
        double_pulse_func = double_sech_pulse
    else:
        single_pulse_func = gaussian_pulse
        double_pulse_func = double_gaussian_pulse
    
    # 时变哈密顿量 [H0, [H_drive, drive_args]]
    H = [H0, [ops['sigma_x'], lambda t, args: 0.5 * (
        double_pulse_func(t, args['omega_0'], args['sigma'], 
                          args['t1'], args['t2'], args['separation']) 
        if args['use_double_pulse'] 
        else single_pulse_func(t, args['omega_0'], args['sigma'], args['t1'])
    )]]
    
    # 创建 Lindblad 算符
    c_ops = create_lindblad_ops(ops, gamma1, gamma_phi, kappa_ph)
    print(f"Lindblad 耗散项:")
    print(f"  - 固有弛豫 γ1 = {gamma1*1000:.2f} MHz")
    print(f"  - 纯退相 γφ = {gamma_phi*1000:.2f} MHz")
    print(f"  - 声子发射 κ_ph = {kappa_ph*1000:.2f} MHz")
    
    # 初始态：qubit 在基态 |g⟩
    psi0 = qutip.basis(N_levels, 0)
    rho0 = psi0 * psi0.dag()
    
    # 可观测量算符
    e_ops = [
        ops['n_op'],           # 激发态布居 P_e
        ops['sigma_z'],        # σ_z 期望值
    ]
    
    # 求解主方程
    print("\n求解 Lindblad 主方程...")
    # 注意：在 QuTiP 5.x 中，需要显式设置 e_ops=None 以返回所有状态
    result = qutip.mesolve(H, rho0, t_list, c_ops, e_ops=None, args=drive_args)
    
    # 从 states 计算期望值
    n_op = ops['n_op']
    sigma_z = ops['sigma_z']
    P_e = np.array([qutip.expect(n_op, rho) for rho in result.states])
    sz_expect = np.array([qutip.expect(sigma_z, rho) for rho in result.states])
    
    # 计算声子发射率
    Gamma_ph = compute_phonon_emission_rate(result.states, kappa_ph, ops)
    
    # 计算累计发射声子数（积分）
    # N_ph(t) = ∫_0^t Γ_ph(τ) dτ
    N_ph = np.cumsum(Gamma_ph) * np.diff(t_list, prepend=t_list[0])
    
    # 计算量子效率
    # 稳态效率：η = N_ph(t) / P_e_max
    # 表示发射的声子数占理论最大发射数的比例
    # 理论最大：如果所有初始激发的能量都转化为声子
    P_e_max = np.max(P_e)
    if P_e_max > 0.001:
        eta = N_ph / P_e_max
    else:
        eta = np.zeros_like(N_ph)
    
    # 限制效率不超过理论最大值（数值误差可能导致略微超限）
    eta = np.clip(eta, 0, 1.0)
    
    # 计算脉冲包络（用于绘图）
    if pulse_type == 'sech':
        if use_double_pulse:
            pulse_envelope = np.array([
                double_sech_pulse(t, omega_0, sigma, t_center, t_center_2, 
                                  pulse_separation if pulse_separation else abs(t_center_2 - t_center))
                for t in t_list
            ])
        else:
            pulse_envelope = np.array([
                sech_pulse(t, omega_0, sigma, t_center) for t in t_list
            ])
    else:  # gaussian
        if use_double_pulse:
            pulse_envelope = np.array([
                double_gaussian_pulse(t, omega_0, sigma, t_center, t_center_2, 
                                      pulse_separation if pulse_separation else abs(t_center_2 - t_center))
                for t in t_list
            ])
        else:
            pulse_envelope = np.array([
                gaussian_pulse(t, omega_0, sigma, t_center) for t in t_list
            ])
    
    print("仿真完成!")
    print(f"  最大激发态布居: {np.max(P_e):.4f}")
    print(f"  最终累计声子数: {N_ph[-1]:.4f}")
    print(f"  峰值量子效率: {np.max(eta):.2%}")
    
    return {
        't_list': t_list,
        'P_e': P_e,
        'sz_expect': sz_expect,
        'Gamma_ph': Gamma_ph,
        'N_ph': N_ph,
        'eta': eta,
        'pulse_envelope': pulse_envelope,
        'rho_final': result.states[-1],
        'result': result
    }


# ============================================================================
# 可视化
# ============================================================================

def plot_results(sim_result, title_suffix="", save_path=None):
    """
    绘制仿真结果
    
    三幅子图：
    1. Qubit 激发态布居 + 脉冲包络
    2. 声子发射量子效率
    3. 累计发射声子数
    
    Parameters
    ----------
    sim_result : dict
        simulate_phonon_emission 的返回结果
    title_suffix : str
        图表标题后缀
    save_path : str, optional
        保存路径
    """
    t = sim_result['t_list']
    P_e = sim_result['P_e']
    eta = sim_result['eta']
    N_ph = sim_result['N_ph']
    pulse = sim_result['pulse_envelope']
    Gamma_ph = sim_result['Gamma_ph']
    
    # 创建图形
    fig = plt.figure(figsize=(10, 9), dpi=DPI)
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.5)
    
    # ========== 子图 1: 布居衰减与脉冲 ==========
    ax1 = fig.add_subplot(gs[0])
    
    # 激发态布居
    ax1.plot(t, P_e, color=COLOR_POPULATION, linewidth=2, label='Excited state population $P_e$')
    ax1.fill_between(t, P_e, alpha=0.3, color=COLOR_POPULATION)
    
    # 脉冲包络（归一化显示）
    pulse_norm = pulse / np.max(pulse) * np.max(P_e) * 0.5
    ax1.plot(t, pulse_norm, '--', color=COLOR_PULSE, linewidth=1.5, alpha=0.7, label='Pulse envelope (scaled)')
    
    ax1.set_ylabel('Population', fontsize=10)
    ax1.set_title(f'Transmon Qubit Population Decay {title_suffix}', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, alpha=0.3)
    
    # ========== 子图 2: 量子效率 ==========
    ax2 = fig.add_subplot(gs[1])
    
    # 限制效率显示范围（避免数值噪声）
    eta_display = np.clip(eta, 0, ETA_MAX * 1.5)
    
    ax2.plot(t, eta_display, color=COLOR_EFFICIENCY, linewidth=2)
    ax2.axhline(y=ETA_MAX, color='gray', linestyle='--', alpha=0.5, label=f'Theoretical max η = {ETA_MAX:.1%}')
    ax2.fill_between(t, eta_display, alpha=0.3, color=COLOR_EFFICIENCY)
    
    ax2.set_ylabel('Quantum Efficiency η', fontsize=10)
    ax2.set_title('Phonon Emission Quantum Efficiency', fontsize=10, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True, alpha=0.3)
    
    # ========== 子图 3: 累计声子数 ==========
    ax3 = fig.add_subplot(gs[2])
    
    ax3.plot(t, N_ph, color=COLOR_CUMULATIVE, linewidth=2, label='Cumulative phonons')
    ax3.fill_between(t, N_ph, alpha=0.3, color=COLOR_CUMULATIVE)
    
    # 添加瞬时发射率（归一化显示）
    Gamma_norm = Gamma_ph / np.max(Gamma_ph) * np.max(N_ph) * 0.3
    ax3.plot(t, Gamma_norm, '--', color='green', linewidth=1.5, alpha=0.6, label='Emission rate (scaled)')
    
    ax3.set_xlabel('Time (ns)', fontsize=10)
    ax3.set_ylabel('Cumulative Phonons $N_{ph}$', fontsize=10)
    ax3.set_title('Cumulative Emitted Phonon Number', fontsize=10, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 调整布局（在添加所有子图后）
    # plt.tight_layout()  # 由于 GridSpec 已控制布局，可省略
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"图形已保存: {save_path}")
    
    plt.show()
    return fig


def plot_comparison(single_result, double_result, save_path=None):
    """
    对比单脉冲和双脉冲（Time-bin）的结果
    
    Parameters
    ----------
    single_result : dict
        单脉冲仿真结果
    double_result : dict
        双脉冲仿真结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=DPI)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    t_single = single_result['t_list']
    t_double = double_result['t_list']
    
    # 单脉冲结果
    ax = axes[0, 0]
    ax.plot(t_single, single_result['P_e'], color=COLOR_POPULATION, linewidth=2)
    ax.fill_between(t_single, single_result['P_e'], alpha=0.3, color=COLOR_POPULATION)
    ax.set_ylabel('$P_e$', fontsize=9)
    ax.set_title('Single Pulse: Population', fontsize=9, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 双脉冲结果
    ax = axes[0, 1]
    ax.plot(t_double, double_result['P_e'], color=COLOR_POPULATION, linewidth=2)
    ax.fill_between(t_double, double_result['P_e'], alpha=0.3, color=COLOR_POPULATION)
    ax.set_ylabel('$P_e$', fontsize=9)
    ax.set_title('Time-bin (Double Pulse): Population', fontsize=9, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 累计声子数对比
    ax = axes[1, 0]
    ax.plot(t_single, single_result['N_ph'], color=COLOR_CUMULATIVE, linewidth=2, label='Single pulse')
    ax.plot(t_double, double_result['N_ph'], color='#8B4513', linewidth=2, label='Double pulse')
    ax.set_xlabel('Time (ns)', fontsize=9)
    ax.set_ylabel('$N_{ph}$', fontsize=9)
    ax.set_title('Cumulative Phonons Comparison', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 效率对比
    ax = axes[1, 1]
    eta_single = single_result['eta']
    eta_double = double_result['eta']
    ax.plot(t_single, eta_single, color=COLOR_EFFICIENCY, linewidth=2, label='Single pulse')
    ax.plot(t_double, eta_double, color='#8B4513', linewidth=2, label='Double pulse')
    ax.axhline(y=ETA_MAX, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (ns)', fontsize=9)
    ax.set_ylabel('η', fontsize=9)
    ax.set_title('Quantum Efficiency Comparison', fontsize=9, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Single vs Time-bin Phonon Emission', fontsize=11, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"对比图已保存: {save_path}")
    
    plt.show()
    return fig


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    from parameters import PULSE_TYPE
    
    pulse_name = "Sech" if PULSE_TYPE == 'sech' else "Gaussian"
    
    print("=" * 70)
    print(f"Transmon Qubit 行波声子发射仿真 - {pulse_name} 脉冲")
    print("Traveling-wave Phonon Emission from Transmon Qubit")
    print("=" * 70)
    
    # 创建输出目录
    import os
    os.makedirs('results', exist_ok=True)
    
    # ========================================================================
    # 仿真 1: 单脉冲驱动
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"仿真 1: 单{pulse_name}脉冲驱动声子发射")
    print("=" * 70)
    
    result_single = simulate_phonon_emission(
        use_double_pulse=False,
        t_center=PULSE_CENTER_1,
        pulse_type=PULSE_TYPE
    )
    
    plot_results(result_single, 
                 title_suffix=f"(Single {pulse_name} Pulse)",
                 save_path=f'results/single_{PULSE_TYPE}_emission.png')
    
    # ========================================================================
    # 仿真 2: 双脉冲 Time-bin 编码
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"仿真 2: Time-bin 编码 - 双{pulse_name}脉冲")
    print("=" * 70)
    
    result_double = simulate_phonon_emission(
        use_double_pulse=True,
        t_center=PULSE_CENTER_1,
        t_center_2=PULSE_CENTER_2,
        pulse_separation=PULSE_CENTER_2 - PULSE_CENTER_1,
        pulse_type=PULSE_TYPE
    )
    
    plot_results(result_double,
                 title_suffix=f"(Time-bin Encoding)",
                 save_path=f'results/timebin_{PULSE_TYPE}_emission.png')
    
    # ========================================================================
    # 对比图
    # ========================================================================
    print("\n" + "=" * 70)
    print("生成对比图...")
    print("=" * 70)
    
    plot_comparison(result_single, result_double,
                    save_path=f'results/comparison_{PULSE_TYPE}.png')
    
    # ========================================================================
    # 结果总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("仿真结果总结")
    print("=" * 70)
    
    print("\n【单脉冲仿真】")
    print(f"  最大激发态布居: {np.max(result_single['P_e']):.4f}")
    print(f"  最终累计声子数: {result_single['N_ph'][-1]:.4f}")
    print(f"  峰值量子效率: {np.max(result_single['eta']):.2%}")
    print(f"  理论最大效率: {ETA_MAX:.2%}")
    
    print("\n【Time-bin 双脉冲仿真】")
    print(f"  最大激发态布居: {np.max(result_double['P_e']):.4f}")
    print(f"  最终累计声子数: {result_double['N_ph'][-1]:.4f}")
    print(f"  峰值量子效率: {np.max(result_double['eta']):.2%}")
    
    # 计算两个脉冲发射的声子数（粗略估计）
    mid_idx = len(T_LIST) // 2
    N_ph_first = result_double['N_ph'][mid_idx]
    N_ph_total = result_double['N_ph'][-1]
    N_ph_second = N_ph_total - N_ph_first
    
    print(f"\n  第一个脉冲发射声子数: ~{N_ph_first:.3f}")
    print(f"  第二个脉冲发射声子数: ~{N_ph_second:.3f}")
    print(f"  总发射声子数: {N_ph_total:.3f}")
    
    print("\n" + "=" * 70)
    print("所有仿真完成！结果保存在 results/ 目录")
    print("=" * 70)
