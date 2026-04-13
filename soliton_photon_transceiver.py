"""
单声子孤子收发系统仿真
=============================
基于 QuTiP 的超导比特-声子波导-超导比特量子传输

系统组成：
---------
1. 发射端：Transmon qubit + IDT 耦合器 → 发射 sech 型单声子
2. 传输通道：LN（铌酸锂）声子波导（色散+非线性+损耗）
3. 接收端：时间反演过程 → 吸收单声子

物理模型：
---------
- 发射/吸收：Lindblad 主方程描述 qubit-声子耦合
- 传播：NLSE 描述波导中的孤子传输
- 时间反演：通过反转传播过程实现吸收

作者: Kimi Code
日期: 2026-04-08
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import qutip
from scipy.integrate import simpson
import matplotlib as mpl

# 使用非交互式后端
mpl.use('Agg')

# ============================================================================
# 物理参数配置
# ============================================================================

class SolitonPhotonParams:
    """
    单声子孤子收发系统参数
    
    基于 LN 声子波导的典型实验参数
    """
    
    # === 超导比特参数 ===
    QUBIT_FREQ = 5.0e9          # Qubit 频率 (Hz)
    ALPHA = -0.3e9              # 非谐性 (Hz)
    GAMMA_1 = 0.001e9           # T1 弛豫 (Hz) -> 1 μs
    GAMMA_PHI = 0.0005e9        # 纯退相 (Hz)
    
    # === IDT 耦合参数 ===
    KAPPA_IDT = 0.02e9          # IDT 发射率 (Hz)，约 20 MHz
    IDT_EFFICIENCY = 0.1        # IDT 效率 10%
    
    # === LN 波导参数 ===
    VG = 3800                   # 群速度 (m/s)，LN 中声子
    WAVEGUIDE_LENGTH = 300e-6   # 波导长度 (300 μm)
    
    # 色散 (5 GHz 附近 LN 的 GVD)
    # β₂ = d²k/dω²，对于 LN 声子波导 β₂ ~ -10⁻²⁰ s²/m
    BETA2 = -1e-20              # 群速度色散 (s²/m)，负值为反常色散
    
    # 非线性系数
    # LN 的三阶非弹性常数导致 γ ~ 10⁻³ W⁻¹m⁻¹
    GAMMA_NL = 1e-3             # 非线性系数 (W⁻¹m⁻¹)
    
    # 传播损耗
    ALPHA_LOSS = 0.5            # 线性损耗 (Np/m)，约 4.3 dB/cm
    
    # === 孤子参数 ===
    PULSE_WIDTH = 1e-9          # 脉冲宽度 τ₀ (1 ns)
    
    # === 仿真参数 ===
    NZ = 200                    # 空间步数
    NT = 1024                   # 时间采样点数
    TIME_WINDOW = 10e-9         # 时间窗口 (10 ns)
    
    def __init__(self):
        # 计算派生参数
        self.z = np.linspace(0, self.WAVEGUIDE_LENGTH, self.NZ)
        self.dz = self.z[1] - self.z[0]
        
        self.t = np.linspace(-self.TIME_WINDOW/2, self.TIME_WINDOW/2, self.NT)
        self.dt = self.t[1] - self.t[0]
        self.df = 1 / self.TIME_WINDOW
        self.f = np.fft.fftfreq(self.NT, self.dt)
        self.omega = 2 * np.pi * self.f
        
        # 计算基态孤子功率
        # N = 1 条件: γP₀τ₀²/|β₂| = 1
        # P₀ = |β₂| / (γτ₀²)
        self.soliton_power = np.abs(self.BETA2) / (self.GAMMA_NL * self.PULSE_WIDTH**2)
        self.soliton_amplitude = np.sqrt(self.soliton_power)
        
        # 传输时间
        self.transit_time = self.WAVEGUIDE_LENGTH / self.VG
        
        # 色散长度
        self.L_D = self.PULSE_WIDTH**2 / np.abs(self.BETA2)
        
        # 非线性长度
        self.L_NL = 1 / (self.GAMMA_NL * self.soliton_power)
        
        print("="*70)
        print("单声子孤子收发系统参数")
        print("="*70)
        print(f"Qubit 频率: {self.QUBIT_FREQ/1e9:.1f} GHz")
        print(f"LN 波导长度: {self.WAVEGUIDE_LENGTH*1e6:.0f} μm")
        print(f"群速度: {self.VG} m/s")
        print(f"传输时间: {self.transit_time*1e9:.2f} ns")
        print(f"\n孤子参数:")
        print(f"  Pulse width tau_0: {self.PULSE_WIDTH*1e9:.2f} ns")
        print(f"  Peak power P_0: {self.soliton_power*1e3:.3f} mW")
        print(f"  Dispersion length L_D: {self.L_D*1e6:.2f} um")
        print(f"  Nonlinear length L_NL: {self.L_NL*1e6:.2f} um")
        print(f"  L_D / L_NL ratio: {self.L_D/self.L_NL:.2f}")
        print(f"\nWaveguide loss: {self.ALPHA_LOSS*4.343:.1f} dB/cm")
        print(f"One-way loss: {self.ALPHA_LOSS*self.WAVEGUIDE_LENGTH*4.343:.2f} dB")
        print("="*70)


# ============================================================================
# 发射端：Qubit 发射 Sech 单声子
# ============================================================================

class PhononTransmitter:
    """
    发射端：Transmon qubit 通过 IDT 发射 sech 型单声子
    """
    
    def __init__(self, params):
        self.params = params
        self.ops = self._create_operators()
        
    def _create_operators(self, N_levels=3):
        """创建 qubit 算符"""
        sigma_minus = qutip.destroy(N_levels)
        sigma_plus = sigma_minus.dag()
        n_op = sigma_plus * sigma_minus
        sigma_x = sigma_plus + sigma_minus
        sigma_z = 2 * n_op - qutip.qeye(N_levels)
        
        return {
            'sigma_minus': sigma_minus,
            'sigma_plus': sigma_plus,
            'n_op': n_op,
            'sigma_x': sigma_x,
            'sigma_z': sigma_z
        }
    
    def sech_pulse(self, t, amplitude, tau, t0=0):
        """Sech 脉冲包络"""
        return amplitude / np.cosh((t - t0) / tau)
    
    def emit_phonon(self, t_center=0, simulate_dynamics=True):
        """
        发射单声子
        
        Returns
        -------
        emitted_wavepacket : dict
            包含发射的声子波包信息
        """
        p = self.params
        ops = self.ops
        
        # 构建哈密顿量（旋转框架）
        H0 = (p.ALPHA / 2) * ops['n_op'] * (ops['n_op'] - 1)
        
        # 驱动脉冲（高斯或 sech 形状）
        # 选择参数使得发射的声子波包近似孤子
        Omega_0 = 0.1e9  # 100 MHz
        sigma = p.PULSE_WIDTH  # 与孤子宽度匹配
        
        # Lindblad 算符
        c_ops = [
            np.sqrt(p.GAMMA_1) * ops['sigma_minus'],           # T1
            np.sqrt(p.GAMMA_PHI/2) * ops['sigma_z'],           # 退相
            np.sqrt(p.KAPPA_IDT) * ops['sigma_minus']          # IDT 发射
        ]
        
        # 初始态 |g>
        psi0 = qutip.basis(3, 0)
        
        if simulate_dynamics:
            # 时间数组
            t_list = np.linspace(-5e-9, 15e-9, 500) + t_center
            
            # 驱动哈密顿量
            drive_args = {
                'sigma_x': ops['sigma_x'],
                'Omega_0': Omega_0,
                'sigma': sigma,
                't0': t_center
            }
            
            H = [H0, [ops['sigma_x'], 
                     lambda t, args: 0.5 * args['Omega_0'] / np.cosh((t-args['t0'])/args['sigma'])]]
            
            # 求解
            result = qutip.mesolve(H, psi0, t_list, c_ops, 
                                  [ops['n_op']], args=drive_args)
            
            # 计算发射的声子数
            P_e = result.expect[0]
            Gamma_emit = p.KAPPA_IDT * P_e
            N_phonon = simpson(Gamma_emit, x=t_list)
            
            print(f"\n发射端仿真:")
            print(f"  Time window: {t_list[0]*1e9:.1f} ~ {t_list[-1]*1e9:.1f} ns")
            print(f"  Max excited state pop: {np.max(P_e):.4f}")
            print(f"  Emitted phonons: {N_phonon:.4f}")
            
            return {
                't_list': t_list,
                'P_e': P_e,
                'Gamma_emit': Gamma_emit,
                'N_phonon': N_phonon,
                'pulse_shape': self.sech_pulse(t_list, np.sqrt(N_phonon), sigma, t_center),
                'tau': sigma,
                'amplitude': np.sqrt(N_phonon)
            }
        else:
            # 简化的发射模型
            # 假设发射效率为 η，发射单声子
            N_phonon = 0.8  # 假设 80% 效率
            return {
                'tau': sigma,
                'amplitude': np.sqrt(N_phonon),
                'N_phonon': N_phonon
            }


# ============================================================================
# 传播：NLSE 孤子传播
# ============================================================================

class SolitonPropagator:
    """
    声子波包在 LN 波导中的孤子传播
    
    使用分步傅里叶方法求解 NLSE
    """
    
    def __init__(self, params):
        self.params = params
        
    def ssfm_propagate(self, u0, z, include_loss=True):
        """
        分步傅里叶传播
        
        Parameters
        ----------
        u0 : array
            初始场分布
        z : array
            传播距离数组
        include_loss : bool
            是否包含损耗
        
        Returns
        -------
        u : 2D array
            场分布 u[z, t]
        """
        p = self.params
        NZ = len(z)
        dz = z[1] - z[0]
        
        # 色散算符（傅里叶域）
        D_op = -p.BETA2 / 2 * p.omega**2
        
        # 增益/损耗
        if include_loss:
            G_op = -p.ALPHA_LOSS / 2
        else:
            G_op = 0
        
        # 存储
        u = np.zeros((NZ, p.NT), dtype=complex)
        u[0, :] = u0
        u_current = u0.copy()
        
        for n in range(1, NZ):
            # 对称分步方法
            # Step 1: 非线性 + 损耗（半步）
            u_half = u_current * np.exp(
                (G_op + 1j * p.GAMMA_NL * np.abs(u_current)**2) * dz/2
            )
            
            # Step 2: 色散（全步）
            u_fft = np.fft.fft(u_half)
            u_fft = u_fft * np.exp(1j * D_op * dz)  # 注意符号
            u_disp = np.fft.ifft(u_fft)
            
            # Step 3: 非线性 + 损耗（半步）
            u_current = u_disp * np.exp(
                (G_op + 1j * p.GAMMA_NL * np.abs(u_disp)**2) * dz/2
            )
            
            u[n, :] = u_current
        
        return u
    
    def create_sech_soliton(self, t, amplitude, tau, t0=0):
        """创建 sech 孤子波包"""
        return amplitude / np.cosh((t - t0) / tau)
    
    def propagate(self, wavepacket, include_loss=True):
        """
        传播声子波包
        
        Parameters
        ----------
        wavepacket : dict
            包含发射的波包信息
        
        Returns
        -------
        propagation_result : dict
            传播结果
        """
        p = self.params
        
        # 创建初始孤子
        if 'pulse_shape' in wavepacket:
            u0 = wavepacket['pulse_shape']
        else:
            u0 = self.create_sech_soliton(
                p.t, 
                wavepacket.get('amplitude', 1),
                wavepacket.get('tau', p.PULSE_WIDTH)
            )
        
        # 传播
        print(f"\nPropagation simulation:")
        print(f"  Propagation distance: {p.WAVEGUIDE_LENGTH*1e6:.0f} um")
        print(f"  Spatial steps: {p.NZ}")
        
        u = self.ssfm_propagate(u0, p.z, include_loss)
        
        # 计算传播特性
        P_input = np.max(np.abs(u0)**2)
        P_output = np.max(np.abs(u[-1, :])**2)
        
        # 计算脉冲宽度变化（FWHM）
        def calc_fwhm(P, t):
            peak = np.max(P)
            half_max = peak / 2
            above_half = t[P > half_max]
            if len(above_half) > 0:
                return above_half[-1] - above_half[0]
            return 0
        
        tau_input = calc_fwhm(np.abs(u0)**2, p.t)
        tau_output = calc_fwhm(np.abs(u[-1, :])**2, p.t)
        
        # 计算能量传输效率
        E_input = simpson(np.abs(u0)**2, x=p.t)
        E_output = simpson(np.abs(u[-1, :])**2, x=p.t)
        transmission_eff = E_output / E_input if E_input > 0 else 0
        
        print(f"  Input peak power: {P_input*1e3:.3f} mW")
        print(f"  Output peak power: {P_output*1e3:.3f} mW")
        print(f"  Peak power change: {10*np.log10(P_output/P_input):.2f} dB")
        print(f"  Input pulse width: {tau_input*1e9:.3f} ns")
        print(f"  Output pulse width: {tau_output*1e9:.3f} ns")
        print(f"  Pulse width change: {(tau_output/tau_input-1)*100:.1f}%")
        print(f"  Energy transmission efficiency: {transmission_eff*100:.1f}%")
        
        return {
            'u': u,
            'u0': u0,
            'P_input': P_input,
            'P_output': P_output,
            'tau_input': tau_input,
            'tau_output': tau_output,
            'transmission_eff': transmission_eff,
            'shape_preserved': abs(tau_output/tau_input - 1) < 0.1  # 变化 < 10%
        }


# ============================================================================
# 接收端：时间反演吸收
# ============================================================================

class PhononReceiver:
    """
    接收端：通过时间反演过程吸收单声子
    
    原理：
    1. 将接收到的波包时间反演
    2. 通过相同的 IDT 耦合器
    3. 优化耦合参数使吸收最大化
    """
    
    def __init__(self, params):
        self.params = params
        self.ops = self._create_operators()
        
    def _create_operators(self, N_levels=3):
        """创建 qubit 算符"""
        sigma_minus = qutip.destroy(N_levels)
        sigma_plus = sigma_minus.dag()
        n_op = sigma_plus * sigma_minus
        sigma_x = sigma_plus + sigma_minus
        sigma_z = 2 * n_op - qutip.qeye(N_levels)
        
        return {
            'sigma_minus': sigma_minus,
            'sigma_plus': sigma_plus,
            'n_op': n_op,
            'sigma_x': sigma_x,
            'sigma_z': sigma_z
        }
    
    def time_reversed_absorption(self, received_wavepacket, t_center=0):
        """
        时间反演吸收
        
        Parameters
        ----------
        received_wavepacket : array
            接收到的波包场分布
        t_center : float
            吸收时间中心
        
        Returns
        -------
        absorption_result : dict
            吸收效率等信息
        """
        p = self.params
        ops = self.ops
        
        # 时间反演：反转时间顺序
        u_time_reversed = np.conj(received_wavepacket[::-1])
        
        # 构建吸收过程（类似发射但时间反演）
        H0 = (p.ALPHA / 2) * ops['n_op'] * (ops['n_op'] - 1)
        
        # 优化的吸收脉冲（时间反演形状）
        # 使用反向 sech 脉冲
        t_list = p.t + t_center
        
        # Lindblad 算符（接收端耦合）
        # 假设接收端耦合系数可以调谐
        kappa_receive = p.KAPPA_IDT  # 使用相同耦合强度
        
        c_ops = [
            np.sqrt(p.GAMMA_1) * ops['sigma_minus'],
            np.sqrt(p.GAMMA_PHI/2) * ops['sigma_z'],
            np.sqrt(kappa_receive) * ops['sigma_minus']  # 吸收通道
        ]
        
        # 初始态 |g>
        psi0 = qutip.basis(3, 0)
        
        # 简化的吸收模型
        # 实际应求解完整的 master equation
        # 这里使用解析估计
        
        # 单声子吸收概率
        # P_abs = 4η / (1+η)²，其中 η = κ_ext / κ_total
        eta_coupling = p.KAPPA_IDT / (p.GAMMA_1 + p.KAPPA_IDT)
        P_abs_theory = 4 * eta_coupling / (1 + eta_coupling)**2
        
        # 考虑传输损耗后的总效率
        transmission_eff = 0.9  # 假设
        total_efficiency = P_abs_theory * transmission_eff
        
        print(f"\nReceiver (time-reversed absorption):")
        print(f"  Coupling efficiency: {eta_coupling:.2%}")
        print(f"  Theoretical single-phonon absorption: {P_abs_theory:.2%}")
        print(f"  Total efficiency (with loss): {total_efficiency:.2%}")
        
        return {
            'u_time_reversed': u_time_reversed,
            'eta_coupling': eta_coupling,
            'P_abs_theory': P_abs_theory,
            'total_efficiency': total_efficiency,
            'absorbed_photons': total_efficiency  # 假设输入单声子
        }


# ============================================================================
# 可视化
# ============================================================================

def plot_complete_system(transmitter, propagator, receiver, results, params):
    """绘制完整系统仿真结果"""
    
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)
    
    t_ns = params.t * 1e9
    z_um = params.z * 1e6
    
    # ========== 发射端 ==========
    ax1 = fig.add_subplot(gs[0, 0])
    if 'emit' in results:
        emit = results['emit']
        ax1.plot(emit['t_list']*1e9, emit['Gamma_emit']/np.max(emit['Gamma_emit']), 
                'b-', linewidth=2, label='Emission rate')
        ax1.plot(emit['t_list']*1e9, emit['P_e'], 'r--', linewidth=2, label='P_e (qubit)')
        ax1.fill_between(emit['t_list']*1e9, emit['Gamma_emit']/np.max(emit['Gamma_emit']), 
                        alpha=0.3, color='blue')
    ax1.set_xlabel('Time (ns)', fontsize=10)
    ax1.set_ylabel('Normalized amplitude', fontsize=10)
    ax1.set_title('Transmitter: Qubit Emission', fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========== 初始波包 ==========
    ax2 = fig.add_subplot(gs[0, 1])
    if 'prop' in results:
        prop = results['prop']
        P0 = np.abs(prop['u0'])**2
        P_out = np.abs(prop['u'][-1, :])**2
        
        ax2.plot(t_ns, P0/np.max(P0), 'b-', linewidth=2, label='Input (z=0)')
        ax2.plot(t_ns, P_out/np.max(P_out), 'r-', linewidth=2, label='Output (z=300μm)')
        ax2.fill_between(t_ns, P0/np.max(P0), alpha=0.3, color='blue')
        ax2.fill_between(t_ns, P_out/np.max(P_out), alpha=0.3, color='red')
    ax2.set_xlabel('Time (ns)', fontsize=10)
    ax2.set_ylabel('Normalized power', fontsize=10)
    ax2.set_title('Wavepacket Shape Preservation', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ========== 传播演化 ==========
    ax3 = fig.add_subplot(gs[1, 0])
    if 'prop' in results:
        prop = results['prop']
        P = np.abs(prop['u'])**2
        T_grid, Z_grid = np.meshgrid(t_ns, z_um)
        im = ax3.pcolormesh(T_grid, Z_grid, P, shading='gouraud', cmap='hot')
        ax3.set_xlabel('Time (ns)', fontsize=10)
        ax3.set_ylabel('Propagation distance (μm)', fontsize=10)
        ax3.set_title('Soliton Propagation in LN Waveguide', fontsize=10, fontweight='bold')
        plt.colorbar(im, ax=ax3, label='Power')
    
    # ========== 频谱演化 ==========
    ax4 = fig.add_subplot(gs[1, 1])
    if 'prop' in results:
        prop = results['prop']
        spec = np.fft.fftshift(np.abs(np.fft.fft(prop['u'], axis=1))**2, axes=1)
        spec_dB = 10 * np.log10(spec + 1e-20)
        spec_dB = spec_dB - np.max(spec_dB)
        
        f_GHz = np.fft.fftshift(params.f) / 1e9
        F_grid, Z_grid = np.meshgrid(f_GHz, z_um)
        im = ax4.pcolormesh(F_grid, Z_grid, spec_dB, shading='gouraud', 
                           cmap='viridis', vmin=-30, vmax=0)
        ax4.set_xlabel('Frequency (GHz)', fontsize=10)
        ax4.set_ylabel('Propagation distance (μm)', fontsize=10)
        ax4.set_title('Spectral Evolution', fontsize=10, fontweight='bold')
        ax4.set_xlim([-5, 5])
        plt.colorbar(im, ax=ax4, label='Power (dB)')
    
    # ========== 接收端 ==========
    ax5 = fig.add_subplot(gs[2, 0])
    if 'receive' in results:
        receive = results['receive']
        # 绘制时间反演波包
        u_tr = receive['u_time_reversed']
        P_tr = np.abs(u_tr)**2
        ax5.plot(t_ns, P_tr/np.max(P_tr), 'g-', linewidth=2)
        ax5.fill_between(t_ns, P_tr/np.max(P_tr), alpha=0.3, color='green')
    ax5.set_xlabel('Time (ns)', fontsize=10)
    ax5.set_ylabel('Normalized power', fontsize=10)
    ax5.set_title('Receiver: Time-Reversed Wavepacket', fontsize=10, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ========== 效率总结 ==========
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    eff_text = ""
    if 'emit' in results:
        eff_text += f"Transmitter:\\n"
        eff_text += f"  Emitted phonons: {results['emit']['N_phonon']:.3f}\\n"
    if 'prop' in results:
        eff_text += f"\\nPropagation:\\n"
        eff_text += f"  Energy transmission: {results['prop']['transmission_eff']*100:.1f}%\\n"
        eff_text += f"  Pulse width change: {(results['prop']['tau_output']/results['prop']['tau_input']-1)*100:.1f}%\\n"
        eff_text += f"  Soliton preserved: {'Yes' if results['prop']['shape_preserved'] else 'No'}\\n"
    if 'receive' in results:
        eff_text += f"\\nReceiver:\\n"
        eff_text += f"  Coupling efficiency: {results['receive']['eta_coupling']*100:.1f}%\\n"
        eff_text += f"  Absorption probability: {results['receive']['P_abs_theory']*100:.1f}%\\n"
        eff_text += f"  Total efficiency: {results['receive']['total_efficiency']*100:.1f}%"
    
    ax6.text(0.1, 0.5, eff_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('Efficiency Summary', fontsize=10, fontweight='bold')
    
    plt.suptitle('Soliton Single-Phonon Transceiver System', 
                fontsize=12, fontweight='bold')
    
    return fig


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*70)
    print("Soliton Single-Phonon Transceiver Simulation")
    print("="*70)
    
    # 初始化参数
    params = SolitonPhotonParams()
    
    # 创建组件
    transmitter = PhononTransmitter(params)
    propagator = SolitonPropagator(params)
    receiver = PhononReceiver(params)
    
    results = {}
    
    # =========================================================================
    # 步骤 1：发射
    # =========================================================================
    print("\n" + "="*70)
    print("Step 1: Transmitter - Qubit emits sech single-phonon")
    print("="*70)
    
    emit_result = transmitter.emit_phonon(t_center=0, simulate_dynamics=True)
    results['emit'] = emit_result
    
    # =========================================================================
    # 步骤 2：传播
    # =========================================================================
    print("\n" + "="*70)
    print("Step 2: Propagation - Soliton transmission in LN waveguide")
    print("="*70)
    
    # 使用发射的波包作为初始条件
    wavepacket = {
        'amplitude': emit_result['amplitude'],
        'tau': emit_result['tau'],
        'N_phonon': emit_result['N_phonon']
    }
    
    prop_result = propagator.propagate(wavepacket, include_loss=True)
    results['prop'] = prop_result
    
    # =========================================================================
    # 步骤 3：接收（时间反演吸收）
    # =========================================================================
    print("\n" + "="*70)
    print("Step 3: Receiver - Time-reversed absorption")
    print("="*70)
    
    received_wavepacket = prop_result['u'][-1, :]
    receive_result = receiver.time_reversed_absorption(
        received_wavepacket, t_center=10e-9
    )
    results['receive'] = receive_result
    
    # =========================================================================
    # 可视化
    # =========================================================================
    print("\n" + "="*70)
    print("Generating visualization")
    print("="*70)
    
    fig = plot_complete_system(transmitter, propagator, receiver, results, params)
    fig.savefig('results/soliton_transceiver_system.png', dpi=150, bbox_inches='tight')
    print("Figure saved: results/soliton_transceiver_system.png")
    
    # =========================================================================
    # 孤子参数总结
    # =========================================================================
    print("\n" + "="*70)
    print("Soliton parameters summary")
    print("="*70)
    print(f"Wavepacket shape: sech(t/tau_0)")
    print(f"Characteristic width tau_0: {params.PULSE_WIDTH*1e9:.2f} ns")
    print(f"Peak power P_0: {params.soliton_power*1e3:.3f} mW")
    print(f"Soliton condition: gamma*P0*tau0^2/|beta2| = {params.GAMMA_NL*params.soliton_power*params.PULSE_WIDTH**2/np.abs(params.BETA2):.2f} (should be 1)")
    print(f"Waveguide length: {params.WAVEGUIDE_LENGTH*1e6:.0f} um")
    print(f"Transit time: {params.transit_time*1e9:.2f} ns")
    
    print("\n" + "="*70)
    print("Simulation completed!")
    print("="*70)
    
    plt.show()
