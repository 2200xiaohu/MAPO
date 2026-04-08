EPM-Q 量化评估体系：计算规程与指标定义
1. 概述 (Overview)
本研究提出的 EPM-Q (Empathy Physics Model - Quantitative) 评估体系旨在提供一套客观、严谨且具有物理可解释性的方法，用于量化评估共情对话模型的综合表现。
为了克服传统封闭式评分（如0-100分制）对模型卓越能力的限制，并解决基准设定的随意性问题，本体系采用了**“基于科学定义的开放式基准指数 (Scientifically-Defined Open Benchmark Index)”**范式。
其核心计算原则是 “基于案例的标准化 (Case-by-Case Normalization)”：鉴于不同测试案例的初始心理赤字（即任务难度）各异，我们拒绝使用单一的全局均值作为基准。相反，我们首先针对每一个具体案例，依据其自身的物理定义计算其特定的基准值，得出该案例的标准化指数；随后，对数据集中所有案例的指数进行宏观平均，得出模型的最终评估结果。

2. 基础科学定义与常数 (Fundamental Scientific Definitions & Constants)
整个量化体系的计算依赖于以下基础定义和科学常数作为锚点。
- 案例特定物理基准：初始赤字半径 ($r_{0,i}$)
  - 定义：对于第 $i$ 个测试案例，其初始时刻用户心理状态向量 $P_{0,i}$ 距离理想平衡原点 $O$ 的欧几里得距离 ($L_2$ 范数)。
  - 公式：$r_{0,i} = \|P_{0,i}\|_2 = \sqrt{c_{0,i}^2 + a_{0,i}^2 + p_{0,i}^2}$
  - 意义：它量化了该特定案例的任务难度，代表了填平该案例心理赤字所需的最小物理有效做功总量。
- 全局数学基准：理论最大强度 ($\rho_{max}$)
  - 定义：在 MDEP 量表体系下（单维得分范围 $[-2, +2]$），理论上可能达到的最大单步行动向量长度。
  - 取值：$\rho_{max} = \sqrt{2^2 + 2^2 + 2^2} = \sqrt{12} \approx \mathbf{3.5}$
  - 意义：它代表了当前测量工具下，模型理论上所能输出的最高单次干预强度极限。
- 全局转换常数：物理-量表转换因子 ($\alpha$)
  - 定义：用于将物理位移需求 ($L_2$范数) 转换为量表原始分需求 ($L_1$范数类似物) 的数学估计系数。
  - 取值：$\mathbf{\alpha \approx 1.5}$
  - 意义：基于向量范数不等式推导出的保守估计值，考虑了真实对话策略的非完美均衡性。
- 全局能量常数：能量转换因子 ($\beta$)
  - 定义：用于设定累积有效能量满分基准的系数。
  - 取值：$\mathbf{\beta \approx 1.2}$
  - 意义：要求模型不仅填平赤字，还需提供额外20%的能量盈余才能获得满分。

3. 核心原始指标定义 (Core Raw Metric Definitions)
本体系基于 EPM 仿真沙盒生成的对话轨迹数据，提取以下三大类核心原始指标。
3.1 结果质量类 (Outcome Quality)
衡量共情干预的最终效果和总量。
- 相对距离改善率 (RDI)：衡量用户最终心理状态相对于初始赤字的改善百分比。公式：$RDI_i = \frac{\|P_{0,i}\| - \|P_{T,i}\|}{\|P_{0,i}\|}$。
- 累积有效能量 ($E_{total}$)：模型在整个对话过程中沿着理想疗愈方向施加的有效物理做功总量。公式：$E_{total,i} = \sum_{t=1}^{T_i} \|\vec{v}_t\| \cos\theta_t$。
- MDEP 总净分 ($S_{net}$)：模型在 MDEP 量表三个维度上获得的累积净得分。公式：$S_{net,i} = \sum_{t=1}^{T_i} \sum_{j \in \{C,A,P\}} v_{t,j}$。
3.2 过程效率类 (Process Efficiency)
衡量达成目标的时间成本和策略直接性。
- 共情密度 ($\rho$)：平均每一轮对话传递的有效共情能量强度。公式：$\rho_i = E_{total,i} / T_i$。
- 平均有效投影分 ($S_{proj}$)：平均每一轮行动向量在理想方向上的投影分量。公式：$S_{proj,i} = \frac{1}{T_i}\sum (\vec{v}_t \cdot \vec{u}_{ideal})$。
- 路径迂回度 ($\tau$)：[新指标] 衡量策略的直接性，定义为实际行动轨迹总长度与起点到终点直线位移之比。公式：$\tau_i = \frac{\sum \|\vec{v}_t\|}{\|P_{0,i} - P_{T,i}\|}$。$\tau \approx 1$ 表示策略高效直接；$\tau \gg 1$ 表示策略迂回试错。
3.3 过程稳定性类 (Process Stability)
衡量交互体验的平滑性、方向正确性和安全性。
- 正能量占比 ($R_{pos}$)：产生正向推动的轮次占总轮次的比例。
- 平均对齐度 (Alignment)：模型干预方向与理想疗愈方向夹角余弦的平均值。
- 表演式惩罚率 ($R_{pen}$)：平均每轮因不当言论而受到的负面评分绝对值总量。

4. 计算规程：从案例数据到最终指数 (Calculation Protocol)
EPM-Q 的计算遵循严格的“三步走”规程。
步骤 1：案例级标准化指数计算 (Case-level Standardization)
对于数据集中的每一个案例 $i$，依据指标物理特性的不同，采用相应的逻辑将其原始值转换为标准化的案例指数 ($Index_i$)。
A. 无界累积型指标 (Unbounded Cumulative Metrics)
逻辑：以该案例特定的物理基准为“100分标尺”，计算实际达成值的倍率。无封顶。
- 累积能量指数：$Index_{E_{tot}, i} = \frac{\max(0, \ E_{total,i})}{\beta \cdot r_{0,i}} \times 100$
- 总净分指数：$Index_{S_{net}, i} = \frac{\max(0, \ S_{net,i})}{\alpha \cdot r_{0,i}} \times 100$
B. 无界强度型指标 (Unbounded Intensity Metrics)
逻辑：以数学理论极限为“100分标尺”，计算逼近极限的程度。无封顶。
- 共情密度指数：$Index_{\rho, i} = \frac{\max(0, \ \rho_i)}{\rho_{max}} \times 100$
- 有效投影指数：$Index_{S_{proj}, i} = \frac{\max(0, \ S_{proj,i})}{\rho_{max}} \times 100$
C. 有界比率型指标 (Bounded Ratio Metrics)
逻辑：依据物理或数学定义的天然确切边界，进行标准的线性映射。
通用公式： $$Index_i = \frac{X_{actual,i} - X_{min}}{X_{max} - X_{min}} \times 100$$
- RDI：边界 $[-100\%, 100\%]$。
- 对齐度：边界 $[-1.0, 1.0]$。
- 正能量占比：边界 $[0\%, 100\%]$。
- 路径迂回度 (反向)：边界 $[3.0, 1.0]$ (1.0为物理最佳，3.0为设定的低效底线)。
- 表演式惩罚率 (反向)：边界 $[3.0, 0.0]$ (0.0为物理最佳，3.0为设定的容忍底线)。
步骤 2：数据集级聚合 (Dataset-level Aggregation)
在完成所有案例的标准化计算后，对模型在整个数据集 ($N$ 个案例) 上的表现进行宏观平均，得到各指标的最终标准化指数 ($\mathbf{\tilde{S}}$)。
$$\mathbf{\tilde{S}_{Metric}} = \frac{1}{N} \sum_{i=1}^{N} Index_{Metric, i}$$
随后，计算三大维度的平均指数：
- $$\mathbf{\tilde{S}_{Outcome}}$$ = Average($\tilde{S}_{RDI}, \tilde{S}_{E_{tot}}, \tilde{S}_{S_{net}}$)
- $$\mathbf{\tilde{S}_{Efficiency}}$$ = Average($\tilde{S}_{\rho}, \tilde{S}_{S_{proj}}, \tilde{S}_{\tau}$)
- $$\mathbf{\tilde{S}_{Stability}}$$ = Average($\tilde{S}_{R_{pos}}, \tilde{S}_{Align}, \tilde{S}_{R_{pen}}$)
步骤 3：合成最终开放基准指数 (Final Synthesis)
应用预设的标准权重体系，将三大维度指数线性合成最终的 EPM 开放基准指数。
$$\mathbf{EPM\text{-}Index} = 0.4 \cdot \mathbf{\tilde{S}_{Outcome}} + 0.2 \cdot \mathbf{\tilde{S}_{Efficiency}} + 0.4 \cdot \mathbf{\tilde{S}_{Stability}}$$

5. 指数解读范式 (Interpretation Paradigm)
本体系输出的是一个开放式基准指数，其解读方式如下：
- Index = 100 (科学基准线)：代表模型的平均表现恰好达成了由任务物理定义和数学理论极限所确定的“标准科学要求”。
- Index > 100 (卓越性体现)：代表模型超越了科学基准。指数值直观反映了其绩效相对于基准的倍率（例如，Index=135 意味着模型在综合效能上达到了标准基准的 1.35 倍），这通常源于模型在累积做功或干预强度上显著超越了基本要求。
- Index < 100 (未达标)：代表模型未能完成基本的科学定义要求。