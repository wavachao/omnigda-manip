# OmniGDA-Manip — 算法工程师简历亮点与硬核面试指南

> **数据一致性声明**：本文档中所有延迟、加速比、显存、mIoU、准确率、AUROC 与物理抓取成功率（GSR）均严格对应 **NVIDIA GeForce RTX 3060 (12 GB)** 实测结果（见 [`results/deploy_benchmark.json`](results/deploy_benchmark.json) 与 [`results/ablation_results.json`](results/ablation_results.json)），确保面试深挖每一项指标均可复现、有据可查。

---

## 一、STAR 法则完整项目阐述（面试口述主线）

- **S（Situation / 背景与痛点）**：
  传统工业/家庭服务机械臂感知链路强依赖大规模标注与重训练，面对产线新增未知异形物体（如新型工装/工具）无法即时上线；而纯 2D 视觉在杂乱遮挡场景下容易混淆类别且缺乏物理尺度感知，纯 3D 几何在相似尺寸物体（如正方体 vs 矩形盒）上判别力骤降，同时 PyTorch 原生多模型串行推理难以满足具身闭环部署的延迟与显存约束。
- **T（Task / 核心目标）**：
  从零设计并工程化落地 **OmniGDA-Manip**——一套**完全免训练（Training-Free）、支持 2~3-Shot 在线热注册未知物体、融合 2D/3D/语言多模态贝叶斯感知与主动视角规划（NBV）**的 Franka Panda 6DoF 闭环抓取系统，打通 **ManiSkill 3 物理仿真 → TensorRT 10.9 FP16 推理加速 → ROS 2 Jazzy 分布式节点解耦** 全栈链路。
- **A（Action / 核心算法与工程落地）**：
  1. **2D 少样本稠密分割（SegmentGDA-Plus）**：基于冻结 DINOv2-ViT-S 多层特征（`L8/10/11`），构建多原型高斯判别分析（Multi-Prototype GDA）并引入 **OAS / Ledoit-Wolf 解析协方差收缩**解决高维小样本（$N \ll D$）矩阵奇异问题；利用 **Sherman-Morrison-Woodbury 秩-$k$ 精度矩阵增量更新**实现免重训练在线注册；结合 $O(HW)$ **快速彩色引导滤波**细化 patch 边缘，并用类别数无关的**前景-背景二元对数似然比（LLR）**解决多类 softmax 稀释导致的漏检问题。
  2. **部分观测感知 3D 多模态 GMDA 融合与开集拒识**：通过置信度加权 RGB-D 反投影 + Open3D SOR 去噪去除边界“飞点”，提取重力对齐 OBB、3D 几何（FPFH）、物理尺寸、OpenCLIP 视觉少样本与零样本文本共 5 路信号；设计**点云完整度评估 + 逆熵门控贝叶斯融合（Entropy-Gated Bayesian Fusion）**自动抑制单视角残缺模态的干扰，并基于可靠性加权马氏距离 $d^2_M$ 实现开集 OOD 拒识。
  3. **主动感知 NBV、双语空间定位与 6DoF 力封闭抓取**：构建 3D 空间场景图支持中英文指令解析（如“抓起最左边的物体”、“put the green box next to the blue bottle”）；高熵/低置信时自动触发 **Next-Best-View（NBV）** 切换辅助相机、Open3D ICP 融合多视角点云并执行逆熵贝叶斯后验锐化；基于置信度加权对跖抓取采样、180° 夹爪对称性 NMS 与 `mplib` 运动规划实现 Franka 物理执行。
  4. **TensorRT 10.9 FP16 部署与 ROS 2 Jazzy 系统解耦**：构建 DINOv2（`518×518`）与动态 batch OpenCLIP（`224×224, B=1..16`）ONNX 导出及 TensorRT 10.9 FP16 引擎构建/推理封装；按 ROS 2 Jazzy 工程规范拆分为 5 个独立包（自定义 msg/srv/action、仿真桥接、在线注册服务、2D/3D 感知、抓取规划与协调器），支持 CycloneDDS + RViz2 实时可视化与 55 项端到端自动化测试。
- **R（Result / 量化成果，RTX 3060 实测）**：
  - **推理加速与精度保真**：TensorRT 10.9 FP16 将 DINOv2-ViT-S（`518×518`）延迟从 `22.44 ms` 降至 **`4.25 ms`（5.28× 加速，235.1 FPS，峰值显存降低 33.7% 至 118 MB，余弦相似度 `0.999980`）**；OpenCLIP ViT-B/32 在 `B=1`/`B=4` 下分别降至 **`1.53 ms`（2.28×，652.1 FPS）** 与 **`1.74 ms`（4.95×，575.1 FPS）**。
  - **少样本感知与开集拒识**：在 ManiSkill 3 20 个随机杂乱场景（160 个物体实例含 40 个未见异形新物体 `cyan_t_tool` / `magenta_cube`）评测中，彩色引导细化使 2D mIoU 提升 **`+12.88%`**（`41.21% → 54.09%`），融合 3D 多模态 GMDA 后 2D mIoU 跃升至 **`73.46%`**，基类与未见少样本新类 Top-1 准确率均达 **`97.5%`**（相比纯 3D 几何新类准确率 `55.0%` 提升 **`+42.5%`**，相比纯 2D 提升 **`+15.0%`**）；开集 OOD 检测 AUROC 达 **`99.48%`**（显著优于纯 2D 的 `60.94%` 与零样本 CLIP 的 `95.38%`）。
  - **免训练在线注册与物理闭环抓取**：新增未知物体 2~3-Shot 在线热注册仅需 **`340~407 ms`**（零梯度更新）；在 18 组跨基类/未知新类/中英文空间指令的 Franka 物理抓取评测中，端到端抓取成功率（GSR）达 **`83.3%`（15/18，基类 100%、新类 83.3%）**，较等权融合（`77.8%`）提升 **`+5.5%`**、较纯 3D 几何（`72.2%`）提升 **`+11.1%`**，并完成空间 Pick-and-Place 闭环验证。

---

## 二、两版一页纸简历精简写法（直接可复用）

### 版本 A：具身智能 / 3D 感知与机械臂操作算法岗（推荐投递机器人、具身智能、抓取操作岗）

**OmniGDA-Manip：免训练多模态 2D-3D 少样本开集具身感知与 6DoF 抓取系统**（核心开发者 | Python / PyTorch / Open3D / ROS 2 Jazzy / ManiSkill 3）
- **2D-3D 少样本免训练感知**：融合 DINOv2-ViT-S 多原型收缩 GDA（SegmentGDA-Plus）与 4 模态部分观测 3D GMDA（几何 FPFH + 物理尺度 + OpenCLIP 视觉/文本）；采用 **OAS 协方差收缩 + Woodbury 秩-$k$ 增量更新**实现未知异形物体 2~3-Shot **`~340–407 ms` 在线热注册**（零重训练），结合快速彩色引导滤波与二元前景 LLR 恢复将 2D mIoU 由 `41.21%` 提至 `73.46%`。
- **不确定性门控融合与主动视角规划（NBV）**：提出点云完整度评估与**逆熵门控贝叶斯融合**，在 ManiSkill 3 20 场景（160 实例）中实现基类/未见新类 Top-1 准确率双 **`97.5%`**（新类较纯 3D 几何 `+42.5%`）、开集 OOD 拒识 AUROC **`99.48%`**；高熵时自动触发 NBV 切换侧视相机并执行 Open3D ICP 点云融合与后验锐化。
- **双语空间接地、6DoF 抓取与 ROS 2 闭环**：构建 3D 场景图支持中英文空间指令解析与置信度感知对跖抓取规划（180° 夹爪对称性 NMS + `mplib` 运动规划），在 18 组 Franka 物理抓取评测中达成 **`83.3%` 端到端 GSR**（基类 100%、新类 83.3%，较等权融合 `+5.5%`）；基于 ROS 2 Jazzy 拆分为 5 个解耦功能包并通过 55 项自动化测试。

### 版本 B：CV / 多模态推理与 TensorRT 部署算法岗（推荐投递多模态感知、边缘部署、推理加速岗）

**OmniGDA-Manip：基于 TensorRT 10.9 与贝叶斯判别分析的实时多模态开集感知部署系统**（核心开发者 | PyTorch / ONNX / TensorRT 10.9 / OpenCLIP / DINOv2 / ROS 2）
- **免微调多模态少样本与开集拒识架构**：在冻结 DINOv2-ViT-S（`L8/10/11`）与 OpenCLIP ViT-B/32 特征空间上建立多原型收缩高斯混合判别分析（OAS 收缩 + Woodbury 在线更新），实现 **`~340–407 ms`** 新类在线注册、**`97.5%`** 少样本分类准确率与 **`99.48%`** 开集 OOD 检测 AUROC（较零样本 CLIP `+4.10%`）。
- **TensorRT 10.9 FP16 引擎优化与数值对齐**：完成 DINOv2（`518×518`）及动态 Batch OpenCLIP（`224×224, B=1..16`）ONNX 导出与 TensorRT 10.9 FP16 静态图构建；在 RTX 3060 上将 DINOv2 加速 **`5.28×`**（`22.44 → 4.25 ms`, `235.1 FPS`, 显存降 `33.7%`, 余弦相似度 `0.999980`），OpenCLIP `B=4` 加速 **`4.95×`**（`8.60 → 1.74 ms`, `575.1 FPS`）。
- **端到端系统剖析与分布式工程落地**：基于真实 640×480 RGB-D 闭环剖析识别出“GPU 推理仅占 <4%、CPU 后处理与 Open3D 去噪占 >92%”的 Amdahl 瓶颈，给出 CUDA 化融合与批量推理优化路径；构建 Docker + `uv` + ROS 2 Jazzy 五包解耦系统与全流程 55 项 CI 级别自动化单测/集成测试。

---

## 三、Top 10 硬核技术面试高频 Q&A

### Q1：为什么在少样本 GDA 中必须引入 OAS / Ledoit-Wolf 协方差收缩？数学原理是什么？
**答**：
- 在 2~3-Shot 场景下，总样本数 $N$ 远小于特征维度 $D$（例如 OpenCLIP 特征 $D=512$，6 类×3-Shot 仅 $N=18$）。此时样本散度矩阵 $S = \sum_{k}\sum_{i \in C_k}(x_i - \mu_k)(x_i - \mu_k)^T$ 的秩至多为 $N - K \ll D$，经验协方差 $\hat{\Sigma} = S/N$ 存在大量零特征值，不可逆且马氏距离 $(x-\mu)^T\hat{\Sigma}^{-1}(x-\mu)$ 会数值爆炸。
- 我们在 `src/omnigda/core/gda_math.py` 中实现了 **Ledoit-Wolf** 与 **OAS（Oracle Approximating Shrinkage，Chen et al. 2010）** 解析收缩估计量：
  $$\Sigma_{\text{shrunk}} = (1 - \alpha)\hat{\Sigma} + \alpha \mu I, \quad \mu = \frac{\operatorname{tr}(\hat{\Sigma})}{D}$$
  其中 OAS 针对高斯分布在高维 $N \ll D$ 情形下给出更紧的闭式最优收缩系数：
  $$\alpha_{\text{OAS}} = \min\!\left(1,\; \frac{(1 - 2/D)\operatorname{tr}(\hat{\Sigma}^2) + \operatorname{tr}(\hat{\Sigma})^2}{(N + 1 - 2/D)\left[\operatorname{tr}(\hat{\Sigma}^2) - \operatorname{tr}(\hat{\Sigma})^2/D\right]}\right)$$
- 与手动调参的固定 Ridge $\lambda I$ 相比，OAS 能根据特征实际各向异性自适应确定 $\alpha \in [0, 1]$，既保证 $\Sigma_{\text{shrunk}}$ 严格对称正定（SPD，配合 `min_eig` 下界与 Cholesky 求逆），又最大限度保留类内协方差相关性。

### Q2：新物体在线热注册如何做到不重算 $O(D^3)$ 矩阵求逆？请推导 Sherman-Morrison-Woodbury 更新。
**答**：
- 在 `OnlineShrinkageGDA.register_online()` 中，新增 $k$ 个样本（或新类）时分为两步：
  1. **Chan 并行轴定理增量更新散度矩阵**：若向已有类追加样本，新旧均值差带来的二阶校正向量 $\delta = (\bar{x}_{\text{new}} - \mu_{\text{old}})\sqrt{\frac{n_{\text{old}} n_{\text{new}}}{n_{\text{old}}+n_{\text{new}}}}$ 与批次残差拼成列矩阵 $U \in \mathbb{R}^{D \times k'}$，使得精确散度增量满足 $\Delta S = U U^T$（新增全新类时 $U$ 即去均值残差矩阵）。
  2. **Sherman-Morrison-Woodbury 秩-$k$ 精度矩阵更新**：设旧精度矩阵 $\Lambda_0 = \Sigma_{\text{old}}^{-1}$，由于总样本数由 $N_{\text{old}} \to N_{\text{new}}$ 缩放且加入低秩项 $\frac{1-\alpha}{N_{\text{new}}} U U^T$，令缩放因子后的基精度为 $\tilde{\Lambda}_0$、低秩因子 $V = \sqrt{\frac{1-\alpha}{N_{\text{new}}}} U$，则：
     $$(\tilde{\Lambda}_0^{-1} + V V^T)^{-1} = \tilde{\Lambda}_0 - \tilde{\Lambda}_0 V \left(I_{k'} + V^T \tilde{\Lambda}_0 V\right)^{-1} V^T \tilde{\Lambda}_0$$
- 只需对 $k' \times k'$ 小矩阵求逆（$k' \le 10$），复杂度从全量重算的 $O(D^3)$ 降至 **$O(D^2 k' + k'^3)$**；同时在 `test_gda_math.py` 中严格验证了 Woodbury 更新结果与全量 Cholesky 重算的数值一致性，实际 RTX 3060 上 2~3-Shot 端到端注册耗时 **`340–407 ms`**（主要耗时在骨干特征提取，线性代数更新仅 <2 ms）。

### Q3：为什么需要“逆熵门控 + 点云完整度”的贝叶斯多模态融合，而不是等权融合？
**答**：
- 机器人单视角 RGB-D 观测下，各模态的信噪比随遮挡和几何对称性剧烈变化：
  - 例 1：物体被部分遮挡时，单视角 2.5D 点云残缺导致 3D FPFH 与 OBB 尺度产生剧烈分布偏移；
  - 例 2：实验表明 `magenta_cube` 与 `green_box` 的 3D 包围盒尺寸近乎一致，纯 3D 几何新类准确率暴跌至 **`55.0%`**，此时 3D 模态后验分布平缓（高归一化香农熵 $H_m \to 1$），而视觉模态后验尖锐（低熵 $H_m \to 0$）。
- 我们在 `entropy_gated_bayesian_fusion()` 中对每个模态 $m$ 计算归一化香农熵 $H_m \in [0, 1]$ 及外部可靠性系数 $\rho_m$（3D 模态取八分象限+方位角覆盖率估计的点云完整度 $\rho_{\text{3D}} \in [0.25, 1.0]$），动态权重为：
  $$w_m \propto \frac{\rho_m}{H_m + \epsilon}, \qquad \log P_{\text{fused}}(y) = \operatorname{log\_softmax}\!\left(\sum_m \tilde{w}_m \log P_m(y)\right)$$
- 实测消融证明：相比等权融合，逆熵门控 + NBV 将闭环 Franka GSR 从 **`77.8%` 提升至 `83.3%`（`+5.5%`）**，并将开集 OOD AUROC 从 `99.17%` 进一步推高至 **`99.48%`**。

### Q4：为什么 TensorRT FP16 将 DINOv2 加速了 5.28×，但端到端 `perceive_frame()` 只加速了 1.08×？如何优化到 >25 FPS？
**答**：
- 这是典型的 **Amdahl 定律（Amdahl's Law）系统瓶颈转移**现象，实测数据非常直观：
  - TensorRT 10.9 FP16 把 DINOv2-ViT-S（`518×518`）从 `22.44 ms` 压到 **`4.25 ms`（5.28×）**，把 OpenCLIP ViT-B/32（`B=1`/`B=4`）压到 **`1.53 ms` / `1.74 ms`（2.28× / 4.95×）**；
  - 但在完整 `perceive_frame()`（`561.7 ms`）中，GPU 神经网络推理已不再是瓶颈：DINOv2 仅占 `seg2d_ms`（`113.6 ms`）的 **3.7%**，其余 `~109 ms` 耗在 CPU NumPy GDA 投影、640×480 双线性上采样与 CPU 彩色引导滤波；`gmda3d_ms`（`445.4 ms`，均摊 9.4 个实例）中，CPU Open3D SOR 去噪、CPU FPFH 计算与逐实例串行调用占绝对大头。
- **下一步落地到 >25 FPS（<40 ms）的明确工程路径**：
  1. 将 GDA 线性判别投影 `W @ F + b` 与盒式引导滤波迁至 PyTorch CUDA Tensor 算子（`109 ms → ~2.5 ms`）；
  2. 将单帧检测出的 $N$ 个目标 crop 合并为单次动态 Batch `B=N` TensorRT OpenCLIP 调用（实测 `B=4` 仅 `1.74 ms`，相比串行调用省掉 70%+ dispatch 与 H2D 拷贝）；
  3. 用 CUDA 核或降采样体素哈希替代 CPU Open3D SOR/FPFH，或引入轻量 TensorRT PointNet++ 几何编码器。

### Q5：为什么在多类（8 类）场景下 2D SegmentGDA 直接阈值化会丢掉目标？你们怎么解决的？
**答**：
- 这是我们在真实 8 类杂乱场景消融中暴露并解决的核心 Badcase：当注册类从 2 类增至 8 类时，9-way softmax（1 背景 + 8 前景类）在前景像素上容易因两类视觉相似而将概率分散为例如 `[bg: 0.02, clsA: 0.48, clsB: 0.46]`，导致单类最大置信度 `< 0.55` 被误判为背景，原始 2D mIoU 暴跌至 `16.04%`。
- 我们在 `OmniGDAAgent._build_foreground_proposal_seg()` 中引入**类别数无关的二元前景-背景对数似然比（Binary FG-vs-BG LLR）**：
  $$p_{\text{fg}}(\mathbf{x}) = \sigma\!\left(\max_{k \ge 1} \ell_k(\mathbf{x}) - \ell_{\text{bg}}(\mathbf{x})\right) \ge 0.70$$
  只要“最佳前景类显著优于背景类”，就保留该像素进入实例候选，再交由 3D 4 模态 GMDA 进行跨模态精准消歧。该设计使 20 个场景 160 个物体的**检测召回率达到 `100.0%`**，并将融合后 2D mIoU 提升至 **`73.46%`**。

### Q6：OpenCLIP ViT-B/32 在 `B=1` 时为什么 `pytorch_fp16`（4.97 ms）反而比 `pytorch_fp32`（3.50 ms）慢？
**答**：
- 这是小模型小 batch GPU Profiling 的经典现象：`ViT-B/32` 在 `224×224` 上仅有 $7\times 7 + 1 = 50$ 个 token，单次 GEMM 计算量极小（memory/launch-bound 而非 compute-bound）。开启 `torch.amp.autocast("cuda", dtype=torch.float16)` 后，PyTorch Eager 模式在每个算子边界插入了额外的 dtype 检查与 FP32↔FP16 cast kernel，kernel launch 开销超过了 TensorCore 节省的算力时间。
- 而在 **TensorRT 10.9 FP16** 中，编译器在离线构建期完成全局常量折叠、QKV/LayerNorm/GELU 算子融合与 FP16 布局固化，消除全部动态 cast 开销，因此 `B=1` 延迟骤降至 **`1.53 ms`（比 PyTorch FP16 快 3.25×）**，`B=4` 也仅增加 `0.21 ms` 至 **`1.74 ms`（比 FP32 快 4.95×）**。

### Q7：主动感知 Next-Best-View（NBV）的触发机制与多视角融合数学流程是什么？
**答**：
- **触发条件**（`OmniGDAAgent.execute_instruction`）：只要满足以下任一不确定性条件且未超最大视角数，即自动触发 NBV 切换至侧视相机 `nbv_camera`：
  1. 语义定位失败（未找到目标或最高匹配分低于阈值）；
  2. 目标归一化分类熵 $H > 0.85$ 或置信度 $p < 0.60$ 或被判定为 OOD；
  3. 单视角点云过稀疏/遮挡导致 6DoF 抓取规划器未找到无碰撞高质量候选（`grasp_out.needs_nbv`）。
- **多视角融合**（`fuse_multiview_detections`）：
  1. 通过 3D 质心近邻贪心匹配（半径 `0.08 m`）关联两视角实例，同时保留视角 2 新发现的被遮挡物体；
  2. 使用 Open3D ICP 精配准 + SOR 去噪（`fuse_multiview_pointclouds`）合并点云并重估重力对齐 OBB，显著提升表面点云密度；
  3. 对分类后验执行**逆熵加权贝叶斯对数后验融合**：$w_v \propto (H_v + 0.05)^{-1}$ 且归一化至 $\sum_v w_v = 2$ 实现真正的贝叶斯置信度锐化。

### Q8：开集 OOD 检测是如何做的？为什么纯 2D 只有 60.9% AUROC，而多模态 GMDA 达到 99.48%？
**答**：
- 纯 2D DINOv2 补丁特征高度表征通用“前景物体性”，未注册的新奇异形物体在 DINOv2 特征空间内仍可能靠近某一已注册物体，导致最大 2D 对数似然的 ID/OOD 分离极弱（`-5.49` vs `-5.00`，AUROC `60.94%`）。
- 零样本 CLIP 仅靠文本余弦距离（`1 - max_cos_sim`），缺乏类内协方差刻度校准，AUROC 为 `95.38%`。
- OmniGDA-Manip 在 `PartialAwareMultimodalGMDA._compute_ood_score()` 中对 3D 几何（FPFH）、物理尺寸（OBB 长宽高与桌高）和 OpenCLIP 视觉少样本三个独立 GMDA 空间分别计算到预测类的最小组件**马氏距离平方** $d^2_{M,m} = (x_m - \mu_{c,m})^T \Lambda_m (x_m - \mu_{c,m})$，再按各模态可靠性 $\rho_m$ 加权聚合。只要未知物体在外观、几何形状或物理尺度任一维度偏离已注册分布，马氏距离就会急剧放大（实测 ID 均值 `13.76` vs OOD 均值 `41.87`，**3.04× 分离度**，AUROC 高达 **`99.48%`**）。

### Q9：6DoF 对跖抓取规划如何处理力封闭评分、碰撞检测与 180° 平行夹爪对称性？
**答**：
- 在 `src/omnigda/grasp/` 中：
  1. **表面法向与对跖采样（`sampler.py`）**：基于目标去噪点云估计一致性外法向，采样满足夹爪行程 $[1.2\text{ cm}, 8.0\text{ cm}]$ 且法向反向对齐（摩擦锥条件 $\mu=0.5$）的对跖点对，同时生成基于 OBB 主轴的顶抓候选兜底。
  2. **置信度感知综合评分与碰撞过滤（`planner.py`, `collision.py`）**：显式构建平行夹爪双指+手掌包围盒，分别检验桌面安全余量（`table_z_margin=0.008 m`）及场景图中所有非目标障碍物点云碰撞；综合分融合力封闭得分、对跖得分与目标感知置信度。
  3. **180° 对称性处理**：平行夹爪绕接近轴 $+Z$ 旋转 $180^\circ$ 物理等价。在 NMS 中使用 `_symmetric_so3_distance_rad` 取 $\min(d(R_a, R_b), d(R_a, R_b R_z(\pi)))$ 避免保留冗余翻转候选；在物理执行层 `FrankaGraspController` 同时尝试原姿态与 $180^\circ$ 翻转姿态（`_flip_gripper_z_180`）并依次重试 Top-3 多样化 NMS 候选，将 Franka 运动规划 IK 失败率降至极低（完整系统 18 组试验 **0 次 IK 规划失败**）。

### Q10：ROS 2 Jazzy 系统的节点拓扑、QoS 设计以及 ManiSkill 3 坐标系对齐有哪些工程细节？
**答**：
- **5 包解耦架构**（`omnigda_full.launch.py` 一键拉起）：
  - `omnigda_interfaces`：定义 7 个 `.msg`（`Object3DArray`, `SceneGraph`, `GraspPoseArray`, `PerceptionMetrics` 等）、3 个 `.srv`（`RegisterObject`, `ListObjects`, `QueryScene`）与 2 个 `.action`（`ExecuteGrasp`, `ExecuteInstruction`）。
  - `omnigda_sim`（`sim_bridge_node`）：发布 `/camera/color/image_raw`、`/camera/depth/image_raw`、`/camera/camera_info`（使用 `BestEffort` + `KeepLast(1)` 传感器 QoS 避免图像队列积压），提供相机切换服务与 `ExecuteGrasp` ActionServer。
  - `omnigda_perception`（`registry_service_node`, `perception_2d_node`, `perception_3d_node`）：在线注册后持久化 2D/3D GDA 记忆并通过 `/gda/memory_updated` 热同步至感知节点；支持 `backend:=tensorrt|onnx|pytorch` 参数切换。
  - `omnigda_manipulation`（`grasp_planner_node`, `coordinator_node`）：解耦抓取规划与任务状态机，发布 RViz2 `MarkerArray`。
- **坐标系严格对齐**：SAPIEN 相机原生为 OpenGL/ROS 习惯，而 `extract_rgbd_from_obs()` 统一取 `intrinsic_cv` 与 `extrinsic_cv` 并求逆得到严格 **OpenCV 光学坐标系（+X 右、+Y 下、+Z 前）到世界系** 的 $T_{w \leftarrow c}$，并将 ManiSkill `int16` 毫米深度统一转为 `float32` 米；`SceneGraphBuilder3D` 严格与 Franka 基座系对齐（$+X$ 远离机械臂、$+Y$ 向左、$+Z$ 向上），彻底消除跨模块坐标系歧义。
