"""
声子非线性传播仿真
====================
求解一维非线性薛定谔方程 (NLSE) 描述声子波包在波导中的传播

物理模型：
---------
∂u/∂z = -β₂/2 ∂²u/∂t² + iγ|u|²u - α/2 u + g/2 u

其中：
- u(z,t): 声子波包包络
- β₂: 群速度色散 (GVD)
- γ: 非线性系数
- α: 传播损耗
- g: 增益 (来自泵浦)

作者: Kimi Code
日期: 2026-04-08
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import simpson
import matplotlib as mpl

# 设置中文字体
mpl.rcParams['font.size'] = 10

# ============================================================================
# 物理参数配置
# ============================================================================

class PhononParams:
    """声子传播物理参数"""
    
    # 泵浦参数 (9 GHz, 15 dBm)
    PUMP_FREQ = 9e9           # 泵浦频率 (Hz)
    PUMP_POWER_DBM = 15       # 泵浦功率 (dBm)
    PUMP_POWER_W = 10**(PUMP_POWER_DBM/10) * 1e-3  # 转换为瓦特
    IDT_EFFICIENCY = 0.10     # IDT 效率 10%
    
    # 声子参数 (5 GHz)
    PHONON_FREQ = 5e9         # 声子频率 (Hz)
    OMEGA_PH = 2 * np.pi * PHONON_FREQ  # 角频率
    
    # 波导参数
    GROUP_VELOCITY = 3800     # 群速度 (m/s)
    WAVEGUIDE_LENGTH = 300e-6 # 波导长度 (300 μm)
    
    # 损耗与增益
    ALPHA_NP = 0.5            # 线性损耗 (Np/m)，约 4.34 dB/cm
    GAIN_DB = 6.5             # 净增益 (dB)，增益略大于损耗
    GAIN_NP = GAIN_DB / 4.343 # 转换为 Np
    
    # 色散参数 (5 GHz 附近)
    # β₂ = -1/vg² * dvg/dω
    # 对于典型声子波导，在 5 GHz 附近 β₂ ~ 10⁻²⁰ s²/m
    BETA2 = 1e-20             # 群速度色散 (s²/m)
    
    # 非线性参数
    # γ = n₂ * ω / (c * A_eff)，对于声子需要重新定义
    # 这里使用有效非线性系数
    GAMMA_NL = 1e-3           # 非线性系数 (1/W/m)
    
    # 仿真参数
    NZ = 500                  # 空间步数
    NT = 2048                 # 时间采样点数
    TIME_WINDOW = 10e-9       # 时间窗口 (10 ns)
    
    def __init__(self):
        # 计算派生参数
        self.z = np.linspace(0, self.WAVEGUIDE_LENGTH, self.NZ)
        self.dz = self.z[1] - self.z[0]
        
        self.t = np.linspace(-self.TIME_WINDOW/2, self.TIME_WINDOW/2, self.NT)
        self.dt = self.t[1] - self.t[0]
        self.df = 1 / self.TIME_WINDOW
        self.f = np.fft.fftfreq(self.NT, self.dt)
        
        # 角频率网格 (用于色散)
        self.omega = 2 * np.pi * self.f
        
        # 有效增益系数 (增益 - 损耗)
        self.g_eff = self.GAIN_NP / self.WAVEGUIDE_LENGTH - self.ALPHA_NP
        
        print("="*60)
        print("声子非线性传播仿真参数")
        print("="*60)
        print(f"泵浦频率: {self.PUMP_FREQ/1e9:.1f} GHz")
        print(f"泵浦功率: {self.PUMP_POWER_DBM} dBm ({self.PUMP_POWER_W*1e3:.2f} mW)")
        print(f"声子频率: {self.PHONON_FREQ/1e9:.1f} GHz")
        print(f"群速度: {self.GROUP_VELOCITY} m/s")
        print(f"波导长度: {self.WAVEGUIDE_LENGTH*1e6:.0f} μm")
        print(f"传播损耗: {self.ALPHA_NP*4.343:.2f} dB/cm")
        print(f"净增益: {self.GAIN_DB} dB (g_eff = {self.g_eff:.2f} Np/m)")
        print(f"GVD (beta_2): {self.BETA2:.2e} s^2/m")
        print(f"非线性系数 (gamma): {self.GAMMA_NL:.2e} 1/W/m")
        print("="*60)


# ============================================================================
# 分步傅里叶方法 (SSFM)
# ============================================================================

def ssfm_propagate(u0, z, t, params, method='symmetric'):
    """
    使用分步傅里叶方法求解 NLSE
    
    方程: ∂u/∂z = -β₂/2 ∂²u/∂t² + iγ|u|²u + (g-α)/2 u
    
    Parameters
    ----------
    u0 : array
        初始场分布 (时间域)
    z : array
        传播距离数组
    t : array
        时间数组
    params : PhononParams
        物理参数
    method : str
        'symmetric' 或 'asymmetric' 分步方法
    
    Returns
    -------
    u : 2D array
        场分布 u[z, t]
    spectrum : 2D array
        频谱分布 spectrum[z, f]
    """
    NZ = len(z)
    NT = len(t)
    dz = z[1] - z[0] if len(z) > 1 else 0
    dt = t[1] - t[0]
    
    # 频率网格
    omega = params.omega
    
    # 色散算符 (傅里叶域)
    # D = -β₂/2 * ω²
    D_op = -params.BETA2 / 2 * omega**2
    
    # 增益/损耗算符
    # G = (g - α) / 2
    G_op = params.g_eff / 2
    
    # 存储结果
    u = np.zeros((NZ, NT), dtype=complex)
    spectrum = np.zeros((NZ, NT), dtype=complex)
    
    u[0, :] = u0
    spectrum[0, :] = np.fft.fft(u0)
    
    u_current = u0.copy()
    
    for n in range(1, NZ):
        if method == 'symmetric':
            # 对称分步傅里叶方法 (SSFM)
            # Step 1: 非线性 + 增益/损耗 (半步)
            u_half = u_current * np.exp((G_op + 1j * params.GAMMA_NL * np.abs(u_current)**2) * dz/2)
            
            # Step 2: 色散 (全步，傅里叶域)
            u_fft = np.fft.fft(u_half)
            u_fft = u_fft * np.exp(D_op * dz)
            u_disp = np.fft.ifft(u_fft)
            
            # Step 3: 非线性 + 增益/损耗 (半步)
            u_current = u_disp * np.exp((G_op + 1j * params.GAMMA_NL * np.abs(u_disp)**2) * dz/2)
            
        else:
            # 非对称方法 (更快但精度较低)
            # 非线性 + 增益
            u_nl = u_current * np.exp((G_op + 1j * params.GAMMA_NL * np.abs(u_current)**2) * dz)
            
            # 色散 (傅里叶域)
            u_fft = np.fft.fft(u_nl)
            u_fft = u_fft * np.exp(D_op * dz)
            u_current = np.fft.ifft(u_fft)
        
        u[n, :] = u_current
        spectrum[n, :] = np.fft.fft(u_current)
    
    return u, spectrum


# ============================================================================
# 初始条件
# ============================================================================

def initial_sech_pulse(t, A, tau_c, t0=0):
    """
    Sech 初始脉冲
    
    u(t) = A * sech((t-t0)/τc)
    
    Parameters
    ----------
    t : array
        时间数组
    A : float
        振幅
    tau_c : float
        特征宽度
    t0 : float
        中心时间
    
    Returns
    -------
    u0 : array
        初始场分布
    """
    return A / np.cosh((t - t0) / tau_c)


def initial_gaussian_pulse(t, A, sigma, t0=0):
    """
    高斯初始脉冲
    
    u(t) = A * exp(-(t-t0)²/(2σ²))
    """
    return A * np.exp(-(t - t0)**2 / (2 * sigma**2))


def calculate_soliton_params(params):
    """
    计算基态孤子参数
    
    基态孤子条件: N = 1
    N² = γP₀τ² / |β₂|
    
    Returns
    -------
    soliton_power : float
        基态孤子峰值功率
    soliton_width : float
        基态孤子宽度
    """
    # 基态孤子: τ₀ = |β₂| / (γP₀)
    # 选择 τ = 1 ns 作为目标宽度
    tau_target = 1e-9  # 1 ns
    
    # P₀ = |β₂| / (γτ²)
    P0 = np.abs(params.BETA2) / (params.GAMMA_NL * tau_target**2)
    
    # 对应的振幅 (|u|² = P)
    A = np.sqrt(P0)
    
    return P0, tau_target, A


# ============================================================================
# 增益模型 (考虑饱和)
# ============================================================================

def gain_saturated(g0, P, P_sat):
    """
    饱和增益模型
    
    g = g0 / (1 + P/P_sat)
    
    Parameters
    ----------
    g0 : float
        小信号增益
    P : float
        瞬时功率
    P_sat : float
        饱和功率
    
    Returns
    -------
    g : float
        饱和增益
    """
    return g0 / (1 + P / P_sat)


# ============================================================================
# 可视化
# ============================================================================

def plot_propagation(u, spectrum, z, t, params, title_suffix=""):
    """
    绘制声子传播结果
    
    Parameters
    ----------
    u : 2D array
        时域场分布
    spectrum : 2D array
        频谱分布
    z : array
        距离数组
    t : array
        时间数组
    params : PhononParams
        参数
    """
    # 转换为实际单位
    z_um = z * 1e6  # μm
    t_ns = t * 1e9  # ns
    f_GHz = params.f / 1e9  # GHz
    
    # 功率 (|u|²)
    P = np.abs(u)**2
    
    # 频谱 (dB)
    spec_dB = 10 * np.log10(np.abs(spectrum)**2 + 1e-20)
    spec_dB = spec_dB - np.max(spec_dB)  # 归一化
    
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.3)
    
    # ========== 时域演化 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    T_grid, Z_grid = np.meshgrid(t_ns, z_um)
    im1 = ax1.pcolormesh(T_grid, Z_grid, P, shading='gouraud', cmap='hot')
    ax1.set_xlabel('Time (ns)', fontsize=10)
    ax1.set_ylabel('Propagation distance (μm)', fontsize=10)
    ax1.set_title(f'Temporal Evolution {title_suffix}', fontsize=10, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Power', fontsize=9)
    
    # ========== 频域演化 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    F_grid, Z_grid = np.meshgrid(f_GHz, z_um)
    im2 = ax2.pcolormesh(F_grid, Z_grid, spec_dB, shading='gouraud', 
                          cmap='viridis', vmin=-40, vmax=0)
    ax2.set_xlabel('Frequency (GHz)', fontsize=10)
    ax2.set_ylabel('Propagation distance (μm)', fontsize=10)
    ax2.set_title(f'Spectral Evolution {title_suffix}', fontsize=10, fontweight='bold')
    ax2.set_xlim([-2, 2])  # 显示中心频率附近
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Power (dB)', fontsize=9)
    
    # ========== 脉冲对比 (输入 vs 输出) ==========
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t_ns, P[0, :], 'b-', linewidth=2, label='Input (z=0)')
    ax3.plot(t_ns, P[-1, :], 'r-', linewidth=2, label=f'Output (z={params.WAVEGUIDE_LENGTH*1e6:.0f}μm)')
    ax3.fill_between(t_ns, P[0, :], alpha=0.3, color='blue')
    ax3.fill_between(t_ns, P[-1, :], alpha=0.3, color='red')
    ax3.set_xlabel('Time (ns)', fontsize=10)
    ax3.set_ylabel('Power', fontsize=10)
    ax3.set_title('Pulse Comparison', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ========== 峰值功率演化 ==========
    ax4 = fig.add_subplot(gs[1, 1])
    P_peak = np.max(P, axis=1)
    P_total = simpson(P, x=t, axis=1)  # 总能量
    
    ax4.plot(z_um, 10*np.log10(P_peak + 1e-20), 'g-', linewidth=2, label='Peak power')
    ax4.plot(z_um, 10*np.log10(P_total + 1e-20), 'm--', linewidth=2, label='Total energy')
    ax4.set_xlabel('Propagation distance (μm)', fontsize=10)
    ax4.set_ylabel('Power/Energy (dB)', fontsize=10)
    ax4.set_title('Power Evolution', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 添加参数文本
    param_text = (f"f = {params.PHONON_FREQ/1e9:.1f} GHz\n"
                  f"v_g = {params.GROUP_VELOCITY} m/s\n"
                  f"L = {params.WAVEGUIDE_LENGTH*1e6:.0f} μm\n"
                  f"Gain = {params.GAIN_DB} dB")
    ax4.text(0.98, 0.98, param_text, transform=ax4.transAxes,
             fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_soliton_formation(u, z, t, params):
    """
    绘制孤子形成过程
    """
    z_um = z * 1e6
    t_ns = t * 1e9
    P = np.abs(u)**2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 选择几个关键位置
    idx = [0, len(z)//4, len(z)//2, 3*len(z)//4, -1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(idx)))
    
    # 时域脉冲演化
    ax = axes[0, 0]
    for i, (ix, c) in enumerate(zip(idx, colors)):
        ax.plot(t_ns, P[ix, :], color=c, linewidth=2, 
                label=f'z = {z_um[ix]:.0f} μm')
    ax.set_xlabel('Time (ns)', fontsize=10)
    ax.set_ylabel('Power', fontsize=10)
    ax.set_title('Pulse Evolution at Different Positions', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 脉冲宽度 (FWHM) 演化
    ax = axes[0, 1]
    fwhm = []
    for i in range(len(z)):
        # 计算 FWHM
        peak = np.max(P[i, :])
        half_max = peak / 2
        above_half = t_ns[P[i, :] > half_max]
        if len(above_half) > 0:
            fwhm.append(above_half[-1] - above_half[0])
        else:
            fwhm.append(0)
    
    ax.plot(z_um, fwhm, 'b-', linewidth=2)
    ax.set_xlabel('Propagation distance (μm)', fontsize=10)
    ax.set_ylabel('Pulse width (ns, FWHM)', fontsize=10)
    ax.set_title('Pulse Width Evolution', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 峰值功率
    ax = axes[1, 0]
    P_peak = np.max(P, axis=1)
    ax.semilogy(z_um, P_peak, 'r-', linewidth=2)
    ax.set_xlabel('Propagation distance (μm)', fontsize=10)
    ax.set_ylabel('Peak power', fontsize=10)
    ax.set_title('Peak Power Evolution', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 时域瀑布图
    ax = axes[1, 1]
    T_grid, Z_grid = np.meshgrid(t_ns, z_um)
    im = ax.pcolormesh(T_grid, Z_grid, P, shading='gouraud', cmap='hot')
    ax.set_xlabel('Time (ns)', fontsize=10)
    ax.set_ylabel('Propagation distance (μm)', fontsize=10)
    ax.set_title('Propagation Waterfall', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Power')
    
    plt.suptitle('Soliton Formation Dynamics', fontsize=12, fontweight='bold')
    
    return fig


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    # 初始化参数
    params = PhononParams()
    
    # 计算孤子参数
    P0_soliton, tau_soliton, A_soliton = calculate_soliton_params(params)
    print(f"\n基态孤子参数:")
    print(f"  峰值功率: {P0_soliton*1e3:.3f} mW")
    print(f"  特征宽度: {tau_soliton*1e9:.2f} ns")
    print(f"  振幅: {A_soliton:.3f}")
    
    # =========================================================================
    # 仿真 1: 基态孤子传播
    # =========================================================================
    print("\n" + "="*60)
    print("仿真 1: 基态孤子传播")
    print("="*60)
    
    # 初始条件: 基态孤子
    u0_soliton = initial_sech_pulse(params.t, A_soliton, tau_soliton, t0=0)
    
    # 传播
    u_soliton, spec_soliton = ssfm_propagate(
        u0_soliton, params.z, params.t, params, method='symmetric'
    )
    
    # 绘图
    fig1 = plot_propagation(u_soliton, spec_soliton, params.z, params.t, 
                            params, title_suffix="(Fundamental Soliton)")
    fig1.savefig('results/soliton_propagation.png', dpi=150, bbox_inches='tight')
    print("图已保存: results/soliton_propagation.png")
    
    fig2 = plot_soliton_formation(u_soliton, params.z, params.t, params)
    fig2.savefig('results/soliton_formation.png', dpi=150, bbox_inches='tight')
    print("图已保存: results/soliton_formation.png")
    
    # =========================================================================
    # 仿真 2: 高功率脉冲 (高阶孤子/非线性效应)
    # =========================================================================
    print("\n" + "="*60)
    print("仿真 2: 高功率脉冲 (N=2 高阶孤子)")
    print("="*60)
    
    # N=2 孤子 (功率是基态的 4 倍)
    A_high = A_soliton * 2
    u0_high = initial_sech_pulse(params.t, A_high, tau_soliton, t0=0)
    
    u_high, spec_high = ssfm_propagate(
        u0_high, params.z, params.t, params, method='symmetric'
    )
    
    fig3 = plot_propagation(u_high, spec_high, params.z, params.t, 
                            params, title_suffix="(N=2 Higher-order Soliton)")
    fig3.savefig('results/higher_order_soliton.png', dpi=150, bbox_inches='tight')
    print("图已保存: results/higher_order_soliton.png")
    
    # =========================================================================
    # 仿真 3: 不同增益对比
    # =========================================================================
    print("\n" + "="*60)
    print("仿真 3: 增益对比")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    gains = [0, 3, 6.5]  # dB
    for g, ax in zip(gains, axes):
        params_temp = PhononParams()
        params_temp.GAIN_DB = g
        params_temp.GAIN_NP = g / 4.343
        params_temp.g_eff = params_temp.GAIN_NP / params_temp.WAVEGUIDE_LENGTH - params_temp.ALPHA_NP
        
        u, spec = ssfm_propagate(u0_soliton, params_temp.z, params_temp.t, 
                                  params_temp, method='symmetric')
        
        P = np.abs(u)**2
        z_um = params_temp.z * 1e6
        t_ns = params_temp.t * 1e9
        
        T_grid, Z_grid = np.meshgrid(t_ns, z_um)
        im = ax.pcolormesh(T_grid, Z_grid, P, shading='gouraud', cmap='hot')
        ax.set_xlabel('Time (ns)', fontsize=10)
        ax.set_ylabel('Propagation distance (μm)', fontsize=10)
        ax.set_title(f'Gain = {g} dB', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('results/gain_comparison.png', dpi=150, bbox_inches='tight')
    print("图已保存: results/gain_comparison.png")
    
    # =========================================================================
    # 结果总结
    # =========================================================================
    print("\n" + "="*60)
    print("仿真结果总结")
    print("="*60)
    
    # 计算压缩因子
    P_input = np.max(np.abs(u_soliton[0, :])**2)
    P_output = np.max(np.abs(u_soliton[-1, :])**2)
    gain_actual = 10 * np.log10(P_output / P_input)
    
    print(f"\n基态孤子:")
    print(f"  输入峰值功率: {P_input*1e3:.3f} mW")
    print(f"  输出峰值功率: {P_output*1e3:.3f} mW")
    print(f"  实际增益: {gain_actual:.2f} dB")
    print(f"  脉冲形状保持: {'良好' if np.abs(gain_actual - params.GAIN_DB) < 1 else '有畸变'}")
    
    print("\n所有结果保存在 results/ 目录")
    print("="*60)
    
    plt.show()
