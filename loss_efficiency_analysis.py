"""
波导损耗对接收效率的影响分析
================================

分析不同传播损耗(alpha)对单声子传输总效率的影响

总效率 = eta_emit × eta_prop × eta_abs
其中 eta_prop = exp(-alpha × L)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

mpl.use('Agg')

# 系统参数
WAVEGUIDE_LENGTH = 300e-6  # 300 um
ETA_EMIT = 0.80            # 发射效率 80%
ETA_ABS_MAX = 0.9994       # 最大吸收效率（临界耦合）

# 耦合效率（影响吸收）
KAPPA_EXT = 0.02e9  # 20 MHz
GAMMA_1 = 0.001e9   # 1 MHz
eta_coupling = KAPPA_EXT / (GAMMA_1 + KAPPA_EXT)
ETA_ABS = 4 * eta_coupling / (1 + eta_coupling)**2

print("="*70)
print("波导损耗对接收效率的影响分析")
print("="*70)
print(f"波导长度: {WAVEGUIDE_LENGTH*1e6:.0f} μm")
print(f"发射效率: {ETA_EMIT*100:.1f}%")
print(f"耦合效率: {eta_coupling*100:.2f}%")
print(f"理论吸收效率: {ETA_ABS*100:.2f}%")
print("="*70)

# 损耗范围 (Np/m)
# 从 0.1 dB/cm (优质波导) 到 20 dB/cm (高损耗)
alpha_dB_cm = np.linspace(0.1, 20, 200)  # dB/cm
alpha_np_m = alpha_dB_cm / 4.343  # 转换为 Np/m

# 计算各项效率
eta_prop = np.exp(-alpha_np_m * WAVEGUIDE_LENGTH)  # 传播效率
eta_total = ETA_EMIT * eta_prop * ETA_ABS            # 总效率

# 关键数据点
key_points = {
    '0.5 dB/cm (LN waveguide)': 0.5,
    '2 dB/cm (Typical)': 2.0,
    '4 dB/cm (Current exp)': 4.0,
    '10 dB/cm (High loss)': 10.0,
}

# 创建图形
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

# ========== 主图：损耗 vs 总效率 ==========
ax1 = fig.add_subplot(gs[0, :])

# 绘制曲线
ax1.semilogy(alpha_dB_cm, eta_total * 100, 'b-', linewidth=2.5, label='Total efficiency')
ax1.semilogy(alpha_dB_cm, eta_prop * 100, 'g--', linewidth=2, alpha=0.7, label='Propagation only')
ax1.axhline(y=ETA_EMIT * ETA_ABS * 100, color='r', linestyle='-.', 
            linewidth=1.5, alpha=0.5, label='Max (no loss)')

# 标注关键点
for label, alpha_val in key_points.items():
    alpha_np = alpha_val / 4.343
    eta_p = np.exp(-alpha_np * WAVEGUIDE_LENGTH)
    eta_t = ETA_EMIT * eta_p * ETA_ABS
    
    ax1.axvline(x=alpha_val, color='gray', linestyle=':', alpha=0.5)
    ax1.scatter([alpha_val], [eta_t * 100], color='red', s=100, zorder=5)
    ax1.annotate(f'{label}\nη={eta_t*100:.1f}%', 
                xy=(alpha_val, eta_t * 100),
                xytext=(alpha_val + 1, eta_t * 100 * 1.5),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7))

ax1.set_xlabel('Waveguide loss (dB/cm)', fontsize=12)
ax1.set_ylabel('Total efficiency (%)', fontsize=12)
ax1.set_title('Waveguide Loss vs. Single-Phonon Transfer Efficiency', 
              fontsize=13, fontweight='bold')
ax1.set_xlim([0, 20])
ax1.set_ylim([1, 100])
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='upper right', fontsize=10)

# 添加参数文本
param_text = (f"Parameters:\n"
              f"  L = {WAVEGUIDE_LENGTH*1e6:.0f} μm\n"
              f"  η_emit = {ETA_EMIT*100:.0f}%\n"
              f"  η_abs = {ETA_ABS*100:.1f}%")
ax1.text(0.98, 0.5, param_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ========== 子图1：线性坐标 ==========
ax2 = fig.add_subplot(gs[1, 0])

ax2.plot(alpha_dB_cm, eta_total * 100, 'b-', linewidth=2)
ax2.fill_between(alpha_dB_cm, eta_total * 100, alpha=0.3, color='blue')

# 标注实验范围
ax2.axvspan(0.5, 4, alpha=0.2, color='green', label='LN range')
ax2.axvline(x=2.17, color='green', linestyle='--', linewidth=2, label='Current (2.17 dB/cm)')

ax2.set_xlabel('Loss (dB/cm)', fontsize=11)
ax2.set_ylabel('Total efficiency (%)', fontsize=11)
ax2.set_title('Linear Scale (Low Loss Region)', fontsize=11, fontweight='bold')
ax2.set_xlim([0, 10])
ax2.set_ylim([0, 100])
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# ========== 子图2：传输距离的影响 ==========
ax3 = fig.add_subplot(gs[1, 1])

# 固定几个损耗值，看距离影响
loss_values = [0.5, 2, 4, 10]  # dB/cm
distances = np.linspace(0, 1000, 200)  # um

colors = plt.cm.viridis(np.linspace(0, 1, len(loss_values)))

for alpha_val, color in zip(loss_values, colors):
    alpha_np = alpha_val / 4.343 * 100  # 转换为 Np/m
    eta_prop_dist = np.exp(-alpha_np * distances * 1e-6)
    eta_total_dist = ETA_EMIT * eta_prop_dist * ETA_ABS
    
    ax3.plot(distances, eta_total_dist * 100, color=color, linewidth=2,
             label=f'{alpha_val} dB/cm')

ax3.axvline(x=300, color='red', linestyle='--', alpha=0.5, label='Current (300 μm)')

ax3.set_xlabel('Waveguide length (μm)', fontsize=11)
ax3.set_ylabel('Total efficiency (%)', fontsize=11)
ax3.set_title('Efficiency vs. Length (Different Loss)', fontsize=11, fontweight='bold')
ax3.set_xlim([0, 1000])
ax3.set_ylim([0, 100])
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9, title='Loss')

plt.suptitle('Loss Impact on Soliton Single-Phonon Transceiver', 
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('results/loss_efficiency_analysis.png', dpi=150, bbox_inches='tight')
print("\n图已保存: results/loss_efficiency_analysis.png")

# ========== 数据表格 ==========
print("\n" + "="*70)
print("不同损耗值下的效率")
print("="*70)
print(f"{'Loss (dB/cm)':<15} {'Loss (Np/m)':<15} {'Propagation':<15} {'Total':<15}")
print("-"*70)

for label, alpha_val in key_points.items():
    alpha_np = alpha_val / 4.343
    eta_p = np.exp(-alpha_np * WAVEGUIDE_LENGTH)
    eta_t = ETA_EMIT * eta_p * ETA_ABS
    print(f"{alpha_val:<15.2f} {alpha_np:<15.4f} {eta_p*100:<15.2f}% {eta_t*100:<15.2f}%")

print("="*70)

# 计算临界损耗（效率降到50%）
target_eff = 0.5
eta_prop_target = target_eff / (ETA_EMIT * ETA_ABS)
alpha_np_critical = -np.log(eta_prop_target) / WAVEGUIDE_LENGTH
alpha_dB_critical = alpha_np_critical * 4.343

print(f"\n临界损耗分析:")
print(f"  当总效率降至 50% 时:")
print(f"  传播效率需要: {eta_prop_target*100:.1f}%")
print(f"  临界损耗: {alpha_dB_critical:.1f} dB/cm ({alpha_np_critical:.1f} Np/m)")

print("="*70)
