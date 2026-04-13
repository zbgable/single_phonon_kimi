"""
声学波导损耗实验数据分析
============================

基于实验数据:
- 损耗: γ = 2951.5 dB/m ≈ 680 Np/m
- Q值: Q_tot ≈ 9k

分析不同波导长度下的传输效率
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

mpl.use('Agg')

# 实验参数
ALPHA_DB_M = 2951.5       # dB/m (实验测量值)
ALPHA_NP_M = 680          # Np/m (近似换算)
Q_FACTOR = 9000           # 品质因数

# 系统参数
ETA_EMIT = 0.80           # 发射效率
ETA_COUPLING = 0.9524     # 耦合效率 (kappa_ext/kappa_tot)
ETA_ABS = 4 * ETA_COUPLING / (1 + ETA_COUPLING)**2  # 吸收效率

print("="*70)
print("声学波导损耗实验数据分析")
print("="*70)
print(f"实验测量损耗: {ALPHA_DB_M:.1f} dB/m ({ALPHA_NP_M:.0f} Np/m)")
print(f"品质因数 Q: {Q_FACTOR}")
print(f"发射效率: {ETA_EMIT*100:.1f}%")
print(f"吸收效率: {ETA_ABS*100:.2f}%")
print("="*70)

# 波导长度范围 (μm)
lengths_um = np.linspace(10, 1000, 500)  # 10 μm 到 1 mm
lengths_m = lengths_um * 1e-6

# 计算损耗 (dB)
loss_dB = ALPHA_DB_M * lengths_m

# 传播效率
eta_prop = 10**(-loss_dB / 10)  # 从 dB 转换
# 或者用 Np: eta_prop = np.exp(-ALPHA_NP_M * lengths_m)

# 总效率
eta_total = ETA_EMIT * eta_prop * ETA_ABS

# 找到关键长度点
def find_length_for_efficiency(target_eff):
    """找到达到目标效率的波导长度"""
    idx = np.where(eta_total >= target_eff)[0]
    if len(idx) > 0:
        return lengths_um[idx[-1]]
    return 0

length_50 = find_length_for_efficiency(0.50)
length_30 = find_length_for_efficiency(0.30)
length_10 = find_length_for_efficiency(0.10)

print(f"\n关键长度点:")
print(f"  效率降至 50% 时: {length_50:.1f} μm")
print(f"  效率降至 30% 时: {length_30:.1f} μm") if length_30 > 0 else print(f"  效率降至 30%: 无法达到")
print(f"  效率降至 10% 时: {length_10:.1f} μm") if length_10 > 0 else print(f"  效率降至 10%: 无法达到")

# 特定长度的数据
print(f"\n不同波导长度下的效率:")
test_lengths = [50, 100, 200, 300, 500, 1000]  # μm
print(f"{'Length (μm)':<15} {'Loss (dB)':<15} {'Propagation':<15} {'Total':<15}")
print("-"*70)

for L_um in test_lengths:
    L_m = L_um * 1e-6
    loss = ALPHA_DB_M * L_m
    eta_p = 10**(-loss / 10)
    eta_t = ETA_EMIT * eta_p * ETA_ABS
    print(f"{L_um:<15} {loss:<15.2f} {eta_p*100:<15.2f}% {eta_t*100:<15.2f}%")

print("="*70)

# 创建图形
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

# ========== 主图：效率 vs 长度 ==========
ax1 = fig.add_subplot(gs[0, :])

# 绘制曲线
ax1.semilogy(lengths_um, eta_total * 100, 'b-', linewidth=2.5, label='Total efficiency')
ax1.semilogy(lengths_um, eta_prop * 100, 'g--', linewidth=2, alpha=0.7, label='Propagation only')

# 最大效率线
eta_max = ETA_EMIT * ETA_ABS * 100
ax1.axhline(y=eta_max, color='r', linestyle='-.', linewidth=1.5, alpha=0.5, 
            label=f'Max (no loss) = {eta_max:.1f}%')

# 标注关键长度
ax1.axvline(x=300, color='purple', linestyle=':', alpha=0.7, linewidth=2)
idx_300 = np.argmin(np.abs(lengths_um - 300))
ax1.scatter([300], [eta_total[idx_300] * 100], color='purple', s=150, zorder=5)
ax1.annotate(f'300 μm\nη={eta_total[idx_300]*100:.1f}%', 
            xy=(300, eta_total[idx_300] * 100),
            xytext=(400, eta_total[idx_300] * 100 * 1.5),
            fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='purple', lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='purple'))

# 标注50%效率点
if length_50 > 0:
    ax1.axhline(y=50, color='orange', linestyle=':', alpha=0.5)
    ax1.axvline(x=length_50, color='orange', linestyle=':', alpha=0.5)
    ax1.scatter([length_50], [50], color='orange', s=100, zorder=5)
    ax1.annotate(f'{length_50:.0f} μm\nfor 50%', 
                xy=(length_50, 50),
                xytext=(length_50 + 100, 70),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='orange'))

ax1.set_xlabel('Waveguide length (μm)', fontsize=12)
ax1.set_ylabel('Efficiency (%)', fontsize=12)
ax1.set_title('Acoustic Waveguide Loss Impact on Single-Phonon Transfer\n' + 
              f'(Experimental: {ALPHA_DB_M:.0f} dB/m, Q ≈ {Q_FACTOR//1000}k)', 
              fontsize=13, fontweight='bold')
ax1.set_xlim([0, 1000])
ax1.set_ylim([1, 100])
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='upper right', fontsize=10)

# 添加实验参数文本
param_text = (f"Experimental Parameters:\n"
              f"  Loss α = {ALPHA_DB_M:.0f} dB/m\n"
              f"  Q ≈ {Q_FACTOR//1000}k\n"
              f"  η_emit = {ETA_EMIT*100:.0f}%\n"
              f"  η_abs = {ETA_ABS*100:.1f}%")
ax1.text(0.98, 0.5, param_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='center', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ========== 子图1：线性坐标（短长度） ==========
ax2 = fig.add_subplot(gs[1, 0])

# 只显示0-200 μm
mask = lengths_um <= 200
ax2.plot(lengths_um[mask], eta_total[mask] * 100, 'b-', linewidth=2.5)
ax2.fill_between(lengths_um[mask], eta_total[mask] * 100, alpha=0.3, color='blue')

# 标注50%点
if length_50 > 0 and length_50 <= 200:
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5)
    ax2.axvline(x=length_50, color='orange', linestyle='--', alpha=0.5)

ax2.set_xlabel('Length (μm)', fontsize=11)
ax2.set_ylabel('Total efficiency (%)', fontsize=11)
ax2.set_title('Short Length Detail (Linear)', fontsize=11, fontweight='bold')
ax2.set_xlim([0, 200])
ax2.set_ylim([0, 100])
ax2.grid(True, alpha=0.3)

# 添加数据标签
for L_um in [50, 100, 150, 200]:
    idx = np.argmin(np.abs(lengths_um - L_um))
    eff = eta_total[idx] * 100
    ax2.annotate(f'{eff:.1f}%', xy=(L_um, eff), xytext=(L_um, eff+5),
                fontsize=8, ha='center')

# ========== 子图2：损耗分解 ==========
ax3 = fig.add_subplot(gs[1, 1])

# 饼图显示300 μm时的效率分解
L_target = 300  # μm
idx_target = np.argmin(np.abs(lengths_um - L_target))
loss_dB_target = ALPHA_DB_M * L_target * 1e-6

# 各环节贡献
eta_p_target = eta_prop[idx_target]
eta_loss = 1 - eta_p_target  # 损耗导致的损失

# 相对效率（相对于最大可能）
rel_emit = ETA_EMIT
rel_prop = eta_p_target
rel_abs = ETA_ABS

categories = ['Emission\n(80%)', 'Propagation\n({:.1f}%)'.format(eta_p_target*100), 
              'Absorption\n({:.1f}%)'.format(ETA_ABS*100)]
values = [rel_emit * 100, rel_prop * 100, rel_abs * 100]
colors = ['#ff9999', '#66b3ff', '#99ff99']

bars = ax3.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Efficiency (%)', fontsize=11)
ax3.set_title(f'Efficiency Breakdown at {L_target} μm\n(Loss = {loss_dB_target:.2f} dB)', 
              fontsize=11, fontweight='bold')
ax3.set_ylim([0, 100])
ax3.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 添加总效率
ax3.text(0.98, 0.98, f'Total: {eta_total[idx_target]*100:.1f}%', 
         transform=ax3.transAxes, fontsize=12, fontweight='bold',
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('results/acoustic_loss_experimental.png', dpi=150, bbox_inches='tight')
print("\n图已保存: results/acoustic_loss_experimental.png")

# ========== 优化建议 ==========
print("\n" + "="*70)
print("优化建议")
print("="*70)

print(f"\n1. 当前实验参数 ({ALPHA_DB_M:.0f} dB/m):")
for L_um in [100, 200, 300, 500]:
    L_m = L_um * 1e-6
    loss = ALPHA_DB_M * L_m
    eta_p = 10**(-loss / 10)
    eta_t = ETA_EMIT * eta_p * ETA_ABS
    status = "[OK]" if eta_t > 0.3 else "[LOW]"
    print(f"   {L_um} μm: 总效率 {eta_t*100:.1f}% {status}")

print(f"\n2. 若要达到 50% 总效率:")
print(f"   最大波导长度: {length_50:.0f} μm")

print(f"\n3. 若要使用 300 μm 波导:")
target_loss_300 = -np.log(0.5 / (ETA_EMIT * ETA_ABS)) / (300e-6)  # Np/m
target_loss_dB = target_loss_300 * 8.686  # dB/m
print(f"   需要将损耗降至: {target_loss_dB:.0f} dB/m")
print(f"   (约为当前的 {target_loss_dB/ALPHA_DB_M*100:.1f}%)")

print("="*70)
