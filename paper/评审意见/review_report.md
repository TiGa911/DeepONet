# Fusion Engineering and Design 审稿报告

**稿件标题**: DeepONet-Based Neural Network for Thomson Scattering Profile Fitting on EAST Tokamak  
**作者**: C.Y. Hu, Q. Zang  
**审稿日期**: 2026年9月9日  
**审稿人**: 匿名审稿人  

---

## 一、总体评价 (Overall Assessment)

本稿件提出了一种基于 DeepONet 的神经网络方法，用于 EAST 托卡马克上 Thomson Scattering (TS) 诊断的剖面拟合（profile fitting），旨在替代传统的七步 mtanh 拟合流程。作者在统一的实验框架下系统比较了五种架构（ProfileNet、BiLSTM、Transformer、CNN-DeepONet、CNN-1D），并进行了消融实验以分离物理约束损失函数与架构归纳偏置的贡献，最后通过 LHW 电流驱动计算展示了剖面差异的下游影响。

**总体而言，本文选题具有实际工程价值，实验设计较为系统，但存在若干方法学和论证层面的重要缺陷，需要实质性修改后方可考虑发表。** 以下将逐一详述。

**建议**: 大修 (Major Revision)

---

## 二、主要意见 (Major Comments)

### M1. 核心局限：神经网络仅学习逼近 mtanh，而非学习真实物理剖面

这是本文最根本的问题。作者在 4.3 节中坦承："The NN models are therefore 'fast approximations to mtanh,' not replacements"（第 192 行）。然而，全文的标题、摘要和引言均以"替代 mtanh 流程"为叙事框架，这造成了严重的定位偏差。

- 所有训练标签均来自 mtanh 拟合结果，这意味着神经网络的上限精度受限于 mtanh 本身的质量。当 mtanh 拟合出现偏差（如 pedestal 起始点选择不当、IQR 阈值误删有效数据点）时，神经网络会忠实地学习这些偏差。
- 作者提到"direct scatter-MSE training performs comparably to mtanh-label training"，但未给出详细数据。**这是关键信息**：如果直接以原始散射点为目标训练，网络的表现如何？这会直接回答"网络是否学到了超越 mtanh 的物理信息"这一核心问题。
- **建议**：(a) 在标题和摘要中明确将网络定位为"mtanh 的快速近似器"而非"替代方法"；(b) 补充 direct scatter-MSE 训练的完整结果（含所有诊断和架构的 MAE 对比表）；(c) 讨论在什么场景下网络可能比 mtanh 更差——例如，当训练数据中的 mtanh 拟合本身存在系统性偏差时。

### M2. 测试集规模过小，统计显著性存疑

测试集仅包含 5 个炮号（shots）、24 个时间点、68 个独立剖面。对于宣称"可替代传统方法"的工作而言，这一测试规模严重不足：

- 表 2 中各架构的 MAE 标准差（±std）很大，且不同架构之间的均值差异往往小于标准差。例如，Te 诊断下 CNN-1D (0.342±0.292) 与 LSTM (0.347±0.194) 之间的差异仅为 0.005 keV，远小于各自的 std。**没有统计检验（如 paired t-test、Wilcoxon signed-rank test）的情况下，无法判断这些差异是否显著。**
- 仅 5 个炮号无法覆盖 EAST 丰富多样的放电场景（不同等离子体电流、密度、加热功率组合、杂质注入实验等）。这严重限制了结论的泛化性。
- 作者排除了 shot 156200（EFIT 覆盖不完整）和 156900 的最后两个时间点（ramp-down），但未说明这些排除是否影响了测试集的代表性。

**建议**：(a) 增加测试集规模，至少覆盖 20-30 个炮号，涵盖更多运行场景；(b) 对所有架构间的关键性能差异提供统计显著性检验；(c) 讨论在更大规模测试中的预期性能变化。

### M3. 缺乏与现有 ML 方法的直接比较

本文引用了 Gaussian Process Regression (GPR) [6,7]、贝叶斯集成数据分析 [8]、CNN 密度剖面重建 [10] 等相关工作，但**完全没有与这些方法进行定量比较**。对于 Fusion Engineering and Design 的读者而言，一个自然的问题是："DeepONet 比 GPR 好在哪里？比 CNN 方法好在哪里？"

- 作者在 4.1 节中类比了 DeepONet 的平滑性与 GPR 的先验，但这只是定性讨论，缺乏定量支撑。
- 既然 CNN-1D 作为控制组已经实现，实现一个 GPR baseline 并不困难（已有成熟的 Python 库如 GPy、GPyTorch）。
- **建议**：至少实现一个 GPR baseline，在相同训练/测试数据上报告 MAE 和推理时间，以便读者判断 DeepONet 方法相对于现有方法的实际优势。

### M4. 消融实验的设计缺陷

消融实验（§3.5）将 **所有** 物理约束同时移除（λ₁=λ₂=λ₃=w_log=0），这导致无法判断每个约束的独立贡献：

- 作者的主要结论——"branch-trunk 结构对静态编码器（ProfileNet、Transformer）是足够的正则化，而 LSTM 仍需要物理约束"——实际上被"同时移除所有约束"的实验设计所混淆。不同的约束可能在 LSTM 和 ProfileNet 上产生不同的效应。
- 例如，L_smooth（二阶导数惩罚）可能对 LSTM 的振荡行为特别重要，而 L_mono（单调性惩罚）可能对 ProfileNet 更重要。同时移除所有约束无法揭示这些差异。
- 此外，消融实验仅在 Te 诊断上进行，未扩展到 ne 和 Ti。Ti 诊断因为使用了双编码器设计，其消融行为可能与 Te 不同。

**建议**：(a) 进行逐约束的消融实验（每次移除一个约束），以识别每个物理约束对不同架构的影响；(b) 将消融实验扩展到 ne 和 Ti 诊断；(c) 补充 CNN-DeepONet 的 physics-off 消融结果。

### M5. LHW 电流驱动 24% 偏差的解读不充分

作者报告 NN 拟合剖面与 mtanh 剖面在 LHW 电流驱动计算中产生约 24% 的偏差（§3.6），但对此数字的解读存在以下问题：

- 24% 是一个相当大的偏差。如果 NN 模型的目标是替代 mtanh 用于下游物理分析，这个偏差是否可接受？作者未给出判断标准。
- 作者承认"Direct validation against experimental current drive measurements is not available"，但没有讨论如何评估哪个方法的 LHW 预测更准确。
- 这一偏差可能源于 pedestal 区域的剖面差异——如 §3.3 所述，NN 在 ρ>0.95 区域存在系统性偏差，而 LHW 功率沉积对边缘密度剖面非常敏感。
- **建议**：(a) 详细分析 24% 偏差的来源（是 Te 还是 ne 的差异主导？在哪个径向区域？）；(b) 讨论在缺乏实验验证的情况下，如何判断 NN 和 mtanh 预测的相对可靠性；(c) 如果可能，补充其他下游物理量的传播分析（如能量约束时间、中子产额等）。

---

## 三、次要意见 (Minor Comments)

### m1. 推理时间对比缺乏基准

作者强调"14 ms forward pass"的速度优势，但未提供 mtanh 流程的实际运行时间作为对比。一个七步参数拟合流程在单核 CPU 上通常也在毫秒量级。读者需要知道速度提升的具体倍数。

**建议**：在同一硬件上测量 mtanh 流程的端到端运行时间（包括 IQR 清洗、spline 拟合、mtanh 优化等），并报告 NN 方法的加速比。

### m2. PCHIP 后处理的影响未量化

作者在 §2.5 中提到"Post-inference, a PCHIP shape-preserving interpolation is applied to enforce strict monotonic decrease and floor-value edge protection. All reported metrics are evaluated on the profiles after this post-processing step." 这意味着报告的 MAE 包含了 PCHIP 后处理的影响，而非原始网络输出。

- 如果 PCHIP 后处理显著改变了剖面形状，那么不同架构之间的性能差异可能部分归因于 PCHIP 对不同输出的修正程度不同，而非架构本身的能力差异。
- **建议**：补充 raw network output（未经过 PCHIP）的 MAE，并与 PCHIP 后的 MAE 对比，量化后处理的贡献。

### m3. Ti 训练数据稀疏的问题

作者在 §4.3 中指出 Ti 训练数据约为 Te 的 43%（148 vs. 342）。考虑到 Ti 使用了双编码器设计且测试集排除了 shot 156900（无 TXCS 数据），Ti 结果的可靠性受到进一步限制。

**建议**：在表 2 中明确标注 Ti 的实际测试样本数（20 个时间点），并讨论样本量不足对 Ti 性能排名的影响。

### m4. CNN-DeepONet 在 Te 和 ne 上表现更差

CNN-DeepONet 控制实验是本文的重要贡献之一，但结果并不完全支持作者的叙事。在 Te 和 ne 上，CNN-DeepONet 均比 CNN-1D 差（Te: 0.342→0.393, ne: 0.067→0.075），仅在 Ti 上表现出显著改善。作者将其解释为"trunk 帮助稀疏诊断但不帮助密集诊断"，但未讨论**为什么 trunk 在密集诊断上反而有害**。

**建议**：分析 trunk 在密集诊断上表现更差的可能原因（过平滑？basis function 数量不足？），并讨论这是否意味着 trunk 设计本身存在不足。

### m5. 引用格式和参考文献完整性问题

- 参考文献 [4] TRANSP and ONETWO 的引用格式不完整，缺少具体的会议论文集页码或 DOI。
- 参考文献 [20] 是会议论文（IAEA FEC contribution），可能不是同行评审的出版物，应注明其状态。
- 参考文献 [22] 的期刊名"Nucl. Eng. Technol."与常见名称"Nuclear Engineering and Technology"略有出入，请核实。该期刊全称应为 Nuclear Engineering and Technology，但缩写是否与 Fusion Engineering 相关领域匹配请核实。
- 部分参考文献缺少 DOI（如 [5]、[17]），建议补充。

### m6. 数学符号和公式

- 公式 (1) 中，Softplus 函数写作 `Softplus`，但 LaTeX 中通常使用 `\operatorname{Softplus}` 以确保正体。
- §2.4 中 L_mono 定义为 `ReLU(dŷ/dρ)`，但这仅惩罚正导数（即非单调递减段）。如果物理上期望剖面严格单调递减，应使用 `ReLU(+dŷ/dρ)` 并明确说明符号约定。当前公式中未使用 `+` 号，含义模糊。

### m7. 写作和语言问题

本文声明使用了 Claude (Anthropic) 进行语言润色，但部分段落仍存在可读性问题：

- 引言（§1）使用了过多长句，建议拆分以提高可读性。
- 摘要中"ablation shows that removing the physics constraints improves ProfileNet (−11.8%) and Transformer (−8.7%) but degrades CNN-1D (+25.8%) and LSTM (+30.6%)"这一句在摘要中过于冗长，建议精简。
- 4.1 节中"Architecture as Regularization"的讨论较为散漫，建议更紧凑地组织论点。

### m8. 图表质量

- 图 1（架构图）对于理解本文至关重要，但当前描述中未提供足够的细节（如各层的具体维度、激活函数）。建议在图中标注关键尺寸。
- 图 3 和 4 的箱线图/柱状图，如果 y 轴标签和单位更清晰，会提高可读性。
- 表 2 中最佳值以粗体标注，但 CNN-DeepONet 在 Te 的 L-mode 行中（0.199）和 Transformer 在 Te 的 L-mode 行中（0.185）均优于 CNN-1D 的 0.206，但 CNN-1D 在"All"行中被标注为最佳。这需要澄清：是"All"行作为主要排名依据，还是各子行独立排名？

### m9. 代码和数据可用性

本文未提供代码仓库链接，仅声明"data available upon reasonable request"。对于一篇以方法学为核心贡献的论文，建议在 GitHub 上公开代码和训练好的模型权重，以增强可复现性。

---

## 四、与 Fusion Engineering and Design 期刊的契合度

本文主题——EAST 托卡马克上 TS 诊断数据的自动化剖面拟合——与 FED 的范围（融合工程、诊断技术、数据处理方法）基本契合。然而，本文在以下方面需要加强与 FED 读者群的连接：

1. **工程实用性**：FED 的读者更关注方法的工程部署可行性、鲁棒性和对装置运行的贡献。建议增加一节，详细讨论将 NN 模型部署到 EAST 实时数据处理流程中的工程挑战（如模型更新策略、异常放电处理、与 MDSplus 的集成等）。
2. **与传统方法的详细对比**：FED 读者期望看到新方法与现有工程方法的全面对比，包括精度、速度、鲁棒性和可维护性等多个维度。
3. **诊断工程视角**：建议从诊断工程的角度讨论该方法对 TS 系统校准、通道故障处理等方面的潜在益处。

---

## 五、推荐意见 (Recommendation)

**大修 (Major Revision)**

### 修改要点总结

| 编号 | 类型 | 内容 |
|------|------|------|
| M1 | 核心 | 明确网络定位为"mtanh 近似器"，补充 direct scatter-MSE 训练结果 |
| M2 | 核心 | 扩大测试集规模（≥20 shots），补充统计显著性检验 |
| M3 | 核心 | 增加 GPR 或至少一种现有 ML 方法的 baseline 对比 |
| M4 | 核心 | 扩展消融实验为逐约束分析，覆盖 ne/Ti 诊断 |
| M5 | 核心 | 深入分析 LHW 24% 偏差的来源和可接受性 |
| m1 | 次要 | 提供 mtanh 流程的实际运行时间作为速度基准 |
| m2 | 次要 | 量化 PCHIP 后处理的独立贡献 |
| m3 | 次要 | 标注 Ti 的实际测试样本量 |
| m4 | 次要 | 分析 trunk 在密集诊断上表现更差的原因 |
| m5 | 次要 | 修正引用格式，补充 DOI |
| m6 | 次要 | 修正数学符号细节 |
| m7 | 次要 | 改进语言表达和段落组织 |
| m8 | 次要 | 改进图表标注和表格排名逻辑 |
| m9 | 次要 | 提供代码仓库链接 |

---

**总体判断**：本文选题具有实际工程价值，五架构系统对比和 CNN-DeepONet 控制实验体现了良好的实验设计意识。但在核心论证（网络仅逼近 mtanh、测试集过小、缺乏与现有方法的比较、消融实验过于粗糙）方面存在实质性缺陷。如果作者能够在修改稿中妥善解决上述主要问题，本文有望达到 FED 的发表标准。

---

*审稿人签名：匿名*  
*日期：2026年9月9日*