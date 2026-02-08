# Morpho Twin 项目原理详解

本文档详细阐述 Morpho Twin 项目的数学原理、控制架构及其核心组件的设计理念。该项目旨在构建一个适应性数字孪生框架，通过实时参数估计、双重控制（Dual Control）和安全滤波器（Safety Filter），实现对不确定系统的安全、高效控制。

## 1. 项目概况与意义

Morpho Twin 是一个集成了参数辨识、最优控制和安全保障的闭环控制系统。其核心目标是在模型参数未知或随时间变化的情况下，不仅能够完成控制任务（如轨迹跟踪），还能主动学习系统参数（Active Learning），并始终保证系统的硬安全约束（Hard Safety Constraints）。

**意义与目的：**
- **自适应性 (Adaptability)**: 通过在线参数估计（MHE），系统能够适应环境变化或自身磨损。
- **主动学习 (Active Learning)**: 控制器不仅利用现有知识进行控制（Exploitation），还通过双重控制策略主动激励系统（Exploration），以获得更好的参数估计。
- **安全性 (Safety)**: 即使在参数不确定性较高的情况下，利用控制障碍函数（CBF）和鲁棒裕度（Robust Margins），也能保证系统状态始终处于安全集合内。

## 2. 核心数学原理

### 2.1 移动视界估计 (Moving Horizon Estimation, MHE)

MHE 用于在线估计系统的状态 $x$ 和参数 $\theta$。不同于卡尔曼滤波（KF）只利用上一时刻的估计，MHE 利用过去一段固定长度的时间窗口内的测量数据进行优化，从而获得更鲁棒的估计结果。

**优化问题：**
在时刻 $t$，MHE 求解以下非线性规划问题（NLP）：

$$
\min_{x_{t-N}, \dots, x_t, \theta} \quad J_{MHE} = \underbrace{\|x_{t-N} - \bar{x}_{t-N}\|_{P_{0}^{-1}}^2 + \|\theta - \bar{\theta}\|_{P_{\theta}^{-1}}^2}_{\text{Arrival Cost}} + \sum_{k=t-N}^{t-1} \underbrace{\|w_k\|_{Q^{-1}}^2}_{\text{Process Noise}} + \sum_{k=t-N}^{t} \underbrace{\|y_k - h(x_k)\|_{R^{-1}}^2}_{\text{Measurement Noise}}
$$

**约束条件：**
$$
\begin{aligned}
x_{k+1} &= f(x_k, u_k, \theta) + w_k, \quad k = t-N, \dots, t-1 \\
x_{\min} &\le x_k \le x_{\max} \\
\theta_{\min} &\le \theta \le \theta_{\max}
\end{aligned}
$$

其中：
- $N$ 为视界长度（Horizon）。
- $\bar{x}_{t-N}, P_0$ 为利用扩展卡尔曼滤波（EKF）更新得到的到达代价（Arrival Cost）先验，包含了视界之前的历史信息。
- $\bar{\theta}, P_{\theta}$ 为参数的先验估计及其协方差。
- $w_k$ 为过程噪声，$y_k$ 为测量值。

**EKF 到达代价更新：**
为了避免丢弃视界之前的信息，项目采用 EKF 来传播到达代价的先验：
$$
\begin{aligned}
\text{预测:} \quad & \hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1}, u_{k-1}, \hat{\theta}) \\
& P_{k|k-1} = A_k P_{k-1|k-1} A_k^T + Q \\
\text{更新:} \quad & K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R)^{-1} \\
& \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (y_k - h(\hat{x}_{k|k-1})) \\
& P_{k|k} = (I - K_k H_k) P_{k|k-1}
\end{aligned}
$$
EKF 的状态和协方差即作为 MHE 下一时刻窗口起始点的先验 $\bar{x}, P_0$。

---

### 2.2 双重控制非线性模型预测控制 (Dual-Control NMPC)

NMPC 负责计算标称控制输入 $u_{nom}$。传统的 NMPC 仅关注跟踪性能（Certainty Equivalence），而双重控制 NMPC 在目标函数中引入了与参数辨识精度相关的项，即费雪信息矩阵（Fisher Information Matrix, FIM）的某种度量。

**优化问题：**
$$
\min_{u_0, \dots, u_{N-1}} \quad J_{NMPC} = J_{tracking} + \lambda_{info} \cdot J_{FIM}
$$

**跟踪代价 (Tracking Cost):**
$$
J_{tracking} = \sum_{k=0}^{N-1} \left( \|x_k - x_{ref}\|_{Q}^2 + \|u_k\|_{R}^2 \right) + \|x_N - x_{ref}\|_{Q_N}^2
$$

**信息代价 (Information Cost):**
为了提高参数的可辨识性，我们需要最大化 FIM。常用的设计准则包括：
- **A-Optimality**: 最小化 FIM 逆矩阵的迹，即 $\min \text{tr}(F^{-1})$。这对应于最小化参数估计误差方差的平均值。
- **D-Optimality**: 最小化 FIM 逆矩阵行列式的对数，即 $\min -\log(\det(F))$。这对应于最小化参数置信椭球的体积。
- **E-Optimality**: 最小化 FIM 最小特征值的倒数，即 $\min \frac{1}{\lambda_{\min}(F)}$。这对应于优化最坏情况方向。

项目代码中实现了上述三种准则的计算逻辑。实际控制实现中，为了保证实时性，采用了基于 **Heuristic Probing**（启发式探测）的策略来近似双重控制效果：

$$
u_{k} = u_{nom, k} + \Delta u_{probe, k}
$$

其中探测信号 $\Delta u_{probe}$ 的幅值与参数的不确定性（协方差迹 $\text{tr}(\Sigma_\theta)$）成正比。当参数不确定性较大时，控制器会主动加入激励信号（如正弦波），以丰富系统的动态响应，从而在后续的 MHE 步骤中获得更精确的参数估计。

---

### 2.3 控制障碍函数安全滤波器 (CBF-QP Safety Filter)

虽然 NMPC 考虑了约束，但由于模型不确定性的存在，NMPC 生成的控制量 $u_{nom}$ 可能不再安全。CBF-QP 作为一个独立的安全层，对 $u_{nom}$ 进行微调，得到最终的安全控制量 $u_{safe}$。

**优化问题 (QP):**
$$
\min_{u, \delta} \quad \frac{1}{2} \|u - u_{nom}\|^2 + \frac{1}{2} \rho \delta^2
$$

**约束条件：**
1. **控制障碍函数约束 (CBF Constraint):**
   $$
   \dot{h}(x, u) + \alpha(h(x)) \ge -\delta - \epsilon_{robust}
   $$
   对于离散时间系统，形式为：
   $$
   h(x_{k+1}) \ge (1-\alpha) h(x_k) - \delta - \epsilon_{robust}
   $$
   其中 $h(x) \ge 0$ 定义了安全集 $\mathcal{C} = \{x \mid h(x) \ge 0\}$。

2. **输入约束:**
   $$
   u_{\min} \le u \le u_{\max}
   $$

3. **松弛变量:** $\delta \ge 0$。

**鲁棒裕度 (Robust Margin) $\epsilon_{robust}$:**
为了应对参数 $\theta$ 的不确定性，CBF 约束被收紧（Tightened）。收紧量 $\epsilon_{robust}$ 依据参数协方差计算：

$$
\epsilon_{robust} = \gamma \cdot \|\nabla_x h\| \cdot \sigma_\theta \cdot \text{margin\_factor}
$$

其中：
- $\sigma_\theta = \sqrt{\text{trace}(\Sigma_\theta)}$ 代表参数估计的总不确定度。
- $\gamma$ 为灵敏度缩放系数。
- $\text{margin\_factor}$ 是由监督层根据系统模式（Mode）动态调整的系数。

这一机制确保了即使真实参数与估计参数存在偏差，只要偏差在协方差椭球覆盖的范围内，系统仍能大概率保持安全。

---

### 2.4 监督与自适应 (Supervision)

监督模块负责监控系统的运行状态，并协调各个组件的工作模式。

**持续激励监控 (Persistence of Excitation, PE):**
为了保证参数收敛，输入信号必须满足持续激励条件。PE Monitor 计算滚动窗口内的 FIM：
$$
F_{rolling} = \sum_{k=t-W}^{t} \phi_k \phi_k^T
$$
并检查其最小特征值：
$$
\lambda_{\min}(F_{rolling}) \ge \lambda_{threshold}
$$
如果 PE 条件不满足，监督器会通知 NMPC 增加探测权重 $\lambda_{info}$ 或切换到更保守的控制模式。

**模式管理 (Mode Management):**
根据 PE 状态和估计误差，系统在不同模式间切换：
- **NORMAL**: 参数收敛，控制激进，Margin 较小。
- **CONSERVATIVE**: 参数不确定，增加 Probing，增大 Safety Margin。
- **SAFE_STOP**: 系统异常或极为不确定，执行安全停机逻辑，最大化 Safety Margin。

## 3. 控制流程图解

1.  **测量 (Measurement)**: 从物理对象（Plant）获取输出 $y$。
2.  **估计 (Estimation)**: MHE 利用 $y$ 和过去的 $u$ 更新状态估计 $\hat{x}$、参数估计 $\hat{\theta}$ 及参数协方差 $\Sigma_\theta$。
3.  **监督 (Supervision)**: 检查 PE 条件，更新系统模式和安全裕度因子。
4.  **控制 (Control)**: NMPC 基于 $(\hat{x}, \hat{\theta})$ 和参考轨迹 $x_{ref}$ 计算标称控制 $u_{nom}$。如果需要（不确定性大），叠加探测信号。
5.  **安全 (Safety)**: CBF-QP 基于 $(\hat{x}, \hat{\theta}, \Sigma_\theta)$ 和 $u_{nom}$ 计算最终安全控制 $u_{safe}$。
6.  **执行 (Actuation)**: 将 $u_{safe}$ 施加于对象。

## 4. 总结

Morpho Twin 通过将 **MHE 的自适应能力**、**双重控制的主动探索机制**以及 **CBF 的鲁棒安全保障**有机结合，形成了一个能够在不确定环境下自我进化、自我保护的先进控制系统。其数学核心在于显式处理了参数的不确定性（协方差 $\Sigma_\theta$），并将其传播到控制目标（FIM Probing）和安全约束（Robust Margin）中。
