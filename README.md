# Transmon Qubit 行波声子发射仿真

基于 QuTiP 的超导量子比特通过 IDT 向行波声子波导发射声子的量子动力学仿真。

## 物理模型

### 核心特点（无 JC 模型）

与传统腔 QED 不同，本仿真**不使用 Jaynes-Cummings 模型**，也**不引入声子谐振子算符**：

- **行波声子环境**：1D 声子波导视为连续谱（continuum），在 Born-Markov 近似下等效为 qubit 的额外耗散通道
- **Lindblad 描述**：声子发射用 qubit 降算符 σ₋ 的耗散直接描述，无需显式声子算符
- **物理图像**：类似于原子自发辐射到自由空间，无需光子/声子数表象

### Hamiltonian

在旋转框架中（ω_d = ω_q）：

```
H = (α/2) n(n-1) + Ω(t) σ_x / 2

其中：
- α：Transmon 非谐性（负值，单位 GHz）
- n = σ₊σ₋：粒子数算符
- Ω(t)：高斯微波脉冲包络
```

### Lindblad 主方程

```
dρ/dt = -i[H, ρ] + L[ρ]

L[ρ] = L_γ1[ρ] + L_γφ[ρ] + L_κph[ρ]

L_γ1[ρ] = γ1 (σ₋ ρ σ₊ - {σ₊σ₋, ρ}/2)    # qubit T1 弛豫（非辐射损耗）
L_γφ[ρ] = (γφ/2) (σz ρ σz - ρ)           # 纯退相
L_κph[ρ] = κph (σ₋ ρ σ₊ - {σ₊σ₋, ρ}/2)   # 向行波声子发射（有用耗散）
```

### 关键区分

| 参数 | 物理意义 | 描述 |
|------|----------|------|
| γ1 | 固有弛豫率 | 能量损耗到非声子环境（如介质损耗）|
| γφ | 纯退相率 | 无能量损耗的相位信息丢失（如电荷噪声）|
| κph | 声子发射率 | qubit → IDT → 行波声子（有用过程）|

### 高斯脉冲驱动

```
Ω(t) = Ω₀ exp(-(t-t₀)² / (2σ²))
```

**Time-bin 编码**：两个分离的高斯脉冲，间隔 Δt >> σ

## 可观测量

1. **激发态布居**：P_e(t) = ⟨σ₊σ₋⟩(t)
2. **声子发射率**：Γ_ph(t) = κph × P_e(t)
3. **累计声子数**：N_ph(t) = ∫₀ᵗ Γ_ph(τ) dτ
4. **量子效率**：η(t) = N_ph(t) / (1 - P_e(t))

## 安装与运行

### 1. 环境准备

```powershell
cd D:\single_phonon_kimi
venv\Scripts\activate
```

### 2. 运行仿真

```powershell
python transmon_phonon_emitter.py
```

### 3. 查看结果

仿真结果保存在 `results/` 目录：
- `single_pulse_emission.png`：单脉冲仿真结果
- `timebin_pulse_emission.png`：Time-bin 双脉冲结果
- `comparison.png`：单脉冲 vs Time-bin 对比

## 参数配置

编辑 `parameters.py` 修改物理参数：

```python
# Qubit 参数
OMEGA_Q = 5.0       # Qubit 频率 (GHz)
ALPHA = -0.3        # 非谐性 (GHz)

# 耗散参数
GAMMA_1 = 0.001     # T1 弛豫率 (GHz)
GAMMA_PHI = 0.0005  # 纯退相率 (GHz)
KAPPA_PH = 0.02     # 声子发射率 (GHz)

# 脉冲参数
SIGMA = 20.0        # 脉冲宽度 (ns)
OMEGA_0 = 0.1       # 驱动幅度 (GHz)
```

**单位系统**：GHz（频率）、ns（时间），自然单位制 ℏ = 1

## 代码结构

```
single_phonon_kimi/
├── venv/                          # Python 虚拟环境
├── parameters.py                  # 物理参数配置
├── transmon_phonon_emitter.py     # 主仿真代码
├── README.md                      # 本说明文档
└── results/                       # 输出结果
    ├── single_pulse_emission.png
    ├── timebin_pulse_emission.png
    └── comparison.png
```

## 物理验证

- **总衰减率**：Γ_total = γ1 + κph + ...
- **理论最大效率**：η_max = κph / (γ1 + κph)
- **能量守恒**：发射声子数 + qubit 剩余能量 ≈ 初始能量

## 参考文献

1. O'Connell et al., Nature 464, 697 (2010) - 量子声学实验奠基工作
2. Gustafsson et al., Science 346, 207 (2014) - 声子波导中的量子发射
3. QuTiP documentation: https://qutip.org/docs/latest/

## 许可

MIT License
