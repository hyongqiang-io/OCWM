# TK-DISA: 时序 Kalman 解耦 Slot 注意力编码器

技术文档 v1.0

## 1. 架构概述

### 1.1 设计目标

- 显式解耦：将物体表示分离为 Position Slot 和 Visual Slot
- 变长支持：动态管理 slot 数量，适应不同场景复杂度
- 时序一致性：通过 Kalman 预测和数据关联处理遮挡，维持跨帧身份
- 组合泛化：Visual Slot 码本化支持外观组合与迁移

### 1.2 核心组件

```text
输入视频帧 I_t -> [DINOv2 编码器] -> F_t ∈ R^(H×W×D)
                     ↓
         [时序 Slot 管理器 SlotManager]
                     ↓
     ┌───────────────┼───────────────┐
     ↓               ↓               ↓
[Position ISA] [Visual ISA]    [数据关联模块]
     ↓               ↓               ↓
 S_p (动力学状态)  S_v (码本化)    匹配 / 创建 / 删除
     ↓               ↓               ↓
     └───────────────┴───────────────┘
                     ↓
         [输出: 变长 Slot 列表]
     {id, state, S_p, S_v_idx, confidence}
```

## 2. 详细架构设计

### 2.1 DINOv2 特征编码器

配置：

- 主干网络：文档原始设计为 DINOv2 ViT-B/14，当前代码默认实现切换为 DINOv2 ViT-S/14
- 参数策略：默认冻结
- 输出特征：优先提取第 4 层和第 11 层 patch token 并拼接
- 空间分辨率：输入图像的 1/14
- 特征维度：当前默认实现为 384 x 2 = 768，可按模型规格自动推导

实现约束：

- 优先调用 `get_intermediate_layers`
- 若底层模型不支持中间层提取，则回退到 `forward_features`
- 输出统一整理为 `[B, D, H, W]`

### 2.2 双分支 D-ISA 编码器

#### 2.2.1 Position ISA 分支

输入：`F_t ∈ R^(B×D×H×W)`

输出：`S_p ∈ R^(B×K×4) = [cx, cy, vx, vy]`

设计要点：

- 固定 `K_max` 个可学习初始 slot
- 位置使用归一化坐标 `[0, 1]`
- 速度初始化为 0
- 通过相对坐标网格编码构建位置敏感注意力
- 采用迭代式 attention + GRU 更新
- 输出 attention mask 供 Visual 分支使用

数值约束：

- attention logits 需要温度缩放
- `cx, cy` 约束在 `[0, 1]`
- `vx, vy` 约束在 `[-v_max, v_max]`

#### 2.2.2 Visual ISA 分支

输入：

- DINO 特征 `F_t`
- Position 分支输出的 `S_p`
- Position attention mask

输出：

- 原始 Visual Slots `S_v ∈ R^(B×K×d_vis)`
- 量化后 Visual Slots `S_v_q`
- 码本索引 `indices`
- `commitment_loss`

设计要点：

- 使用 Position attention mask 对特征做加权聚合
- 仅阻断 mask 的梯度，避免 Visual 分支反向影响 Position 分支
- 通过 GRU 迭代更新 Visual slots
- 使用 EMA Vector Quantization 完成码本化

默认配置：

- `d_vis = 256`
- `codebook_size = 512`
- `decay = 0.99`

### 2.3 时序 Slot 管理器

核心职责：

- 维护跨帧 slot 状态
- 数据关联
- 动力学预测与更新
- 生命周期管理

状态机：

- `ACTIVE`
- `OCCLUDED`
- `INACTIVE`

更新流程：

1. 对已有 track 进行动力学预测
2. 使用位置代价和外观代价进行匹配
3. 更新已匹配 track
4. 对未匹配旧 track 进入遮挡维持或删除流程
5. 对未匹配新检测创建新 track

关联代价：

```text
cost = position_cost + 0.5 * visual_mismatch_cost
```

其中：

- `position_cost = ||predicted_pos - detected_pos|| / pos_threshold`
- `visual_mismatch_cost ∈ {0, 1}`

## 3. 训练流程

### 3.1 阶段一：单帧 D-ISA 预训练

目标：

- 学习有效的解耦表示
- 训练 Visual 码本

数据：

- COCO
- ImageNet
- CLEVR

建议损失：

```python
def compute_loss(
    features,
    recon_features,
    commit_loss,
    s_p_pred,
    s_p_gt=None,
    loss_swap=None,
    alpha=1.0,
    beta=1.0,
    gamma=0.1,
):
    loss_recon = F.mse_loss(recon_features, features)
    loss_vq = commit_loss

    if s_p_gt is not None:
        loss_pos = F.mse_loss(s_p_pred[:, :, :2], s_p_gt)
    else:
        loss_pos = torch.zeros((), device=features.device)

    if loss_swap is None:
        loss_swap = torch.zeros((), device=features.device)

    total_loss = loss_recon + alpha * loss_vq + beta * loss_pos + gamma * loss_swap
    return total_loss
```

训练配置：

- batch size: 32
- optimizer: AdamW
- learning rate: `1e-4`
- 训练步数：100k
- `K_max = 16`

### 3.2 阶段二：时序一致性训练

目标：

- 学习跨帧一致的 Visual 码本分配
- 学习稳定动力学

数据：

- YouTube-VOS
- DAVIS
- 自动驾驶视频序列

建议约束：

- 视觉码本一致性
- 速度平滑
- 遮挡恢复鲁棒性

训练技巧：

- 随机遮挡模拟
- 课程学习：短序列到长序列
- 教师强制到自回归关联的过渡

### 3.3 阶段三：端到端微调

优化策略：

- DINOv2：冻结
- D-ISA 编码器：`1e-5`
- SlotManager 参数：`1e-4`

建议评估指标：

- DINO 特征重建误差
- 码本使用率
- MOTA / IDF1
- 交换实验下的解耦度

## 4. 推理流程

### 4.1 单帧推理

流程：

1. 图像输入 DINOv2 编码器
2. Position ISA 生成位置 slot 和 attention mask
3. Visual ISA 在 position mask 约束下生成视觉 slot
4. 对 visual slot 做 VQ 码本索引
5. 基于 attention 峰值或对象性分数做前景过滤

输出格式：

```python
[
    {
        "position": [cx, cy, vx, vy],
        "visual_code": code_index,
        "confidence": confidence,
    }
]
```

### 4.2 视频推理

流程：

1. 逐帧提取 D-ISA 检测结果
2. 使用 SlotManager 执行跨帧关联
3. 在未匹配情况下进入遮挡预测或新建 slot
4. 输出每帧变长 slot 列表

## 5. 关键实现细节

### 5.1 数值稳定性

1. Attention 温度缩放
2. 速度范围裁剪
3. 码本死码监控与重置

### 5.2 内存优化

1. D-ISA 迭代使用 checkpoint
2. 训练阶段开启 AMP
3. 视频推理阶段使用动态 batching

### 5.3 调试建议

1. 可视化 Position attention mask
2. 可视化 Visual code 聚类分布
3. 跟踪跨帧 ID 一致性
4. 监控平均 slot 数、遮挡恢复率、码本覆盖率

## 6. 项目落地约定

当前仓库中的工程实现放置于：

- `docs/TK-DISA.md`
- `module/slot_encoder/`
- `module/dynamic/`
- `script/`

首轮开发范围包括：

- DINOv2 编码器封装
- Position ISA
- Visual ISA
- EMA VQ 码本
- SlotManager 与动态状态管理
- 顶层 TK-DISA 编码器组合模块
- 独立 `script/` 包中的训练 / 推理 / 评估 pipeline

## 7. 总结

TK-DISA 的核心思想是把几何动力学和外观表征显式拆开，再通过时序管理维持对象身份。该架构适合复杂场景下的检测、追踪、遮挡恢复和组合生成任务。首轮工程实现优先完成编码器与时序管理骨架，为后续训练、解码器和下游任务接口预留稳定结构。

## 8. 当前工程实现

当前仓库已补齐以下工程模块：

- `module/slot_encoder/model.py`：完整 `TKDISAModel`，负责编码、解码与 detection 输出
- `module/slot_encoder/decoder.py`：基于 Position Slot 和量化 Visual Slot 的特征空间 decoder
- `module/dynamic/slot_manager.py`：跨帧状态管理、Kalman 风格运动预测与关联
- `script/static.py`：单帧训练 / 推理 / 评估 pipeline，以及静态损失
- `script/dynamic.py`：视频训练 / 推理 / 评估 pipeline，以及时序损失
- `script/common.py`：AMP 与指标转换等共享工具

## 9. GPU 适配说明

当前工程实现已适配 CUDA 设备：

- 若检测到可用 GPU，训练与推理 pipeline 默认优先使用 `cuda`
- 训练 pipeline 默认开启 AMP 混合精度
- 推理 pipeline 默认开启 AMP 混合精度
- DINOv2 封装在无 CUDA 场景下会自动禁用 xFormers，避免 CPU 前向报错

## 5. 当前简化版动力学方案（研究讨论稿）

说明：

- 本节记录 2026-03 的最新设计讨论，目标是为后续世界模型预留接口。
- 本节是对上文 `SlotManager + 显式状态机` 思路的重构，不直接否定已有实现，但代表当前更优先的研究方向。
- 当前原则是“先从简”，优先搭建一个可训练、可扩展的对象中心 belief-state filter。

### 5.1 核心定位

新的动力学部分不再把时序问题理解为传统 tracking 状态机，而是理解为：

- 静态编码器产生单帧对象证据
- 动力学模块维护跨时间的对象信念状态
- 当前帧观测用于校正先验状态
- 全局分支提供场景级条件，而不是替代 slot 成为第二套主状态系统

对应的职责边界：

- `slot` 是主状态，负责对象级持久性、遮挡恢复、重识别和交互
- `global` 是条件状态，负责提供场景上下文、边界变化和后验门控
- `graph` 只提供 slot-slot 关系上下文，不直接改写状态

### 5.2 Observation 与 State 分离

#### 5.2.1 Slot Observation

静态编码器输出的是 `slot observation`，而不是完整的动力学状态。

定义：

```text
O_t^{s,i} = (p_obs_t^i, q_obs_t^i, c_obs_t^i)
```

其中：

- `p_obs_t^i`：相对位置观测
- `q_obs_t^i`：视觉码观测，建议用 codebook logits / belief，而不是只保留单一 index
- `c_obs_t^i`：观测质量或支持强度

当前项目对应关系：

- `p_obs` 可由 Position ISA 提供
- `q_obs` 由基于 VQ codebook 的视觉分支提供
- `c_obs` 不再只等同于 attention peak，建议由多种单帧质量信号组合而成

#### 5.2.2 Slot State

动力学系统维护的是 `slot state`：

```text
s_t^i = (p_t^i, a_t^i, e_t^i, u_t^i, h_t^i)
```

其中：

- `p_t^i`：持久几何状态
- `a_t^i`：持久外观状态，当前建议直接作为 persistent code belief
- `e_t^i`：existence logit
- `u_t^i`：uncertainty
- `h_t^i`：slot memory / residual hidden state

当前第一版建议：

- `p_t` 与 `p_obs_t` 先保持同维
- `a_t` 直接建模为对视觉码本的时序 belief
- 不额外引入连续 appearance latent

#### 5.2.3 Global Observation 与 Global State

全局观测建议直接来自原图编码，而不是 slot 后处理结果：

```text
O_t^g = E_g(x_t) = (f_t^g, o_t^g)
```

其中：

- `f_t^g`：dense feature map / token
- `o_t^g`：pooled global observation

全局状态第一版保持紧凑：

```text
g_t
```

设计原则：

- 不过早细拆语义字段
- 只承担 prior modulation 和 posterior gating
- 避免演化成与 slot 对称的第二套复杂世界模型

### 5.3 U 型时序更新顺序

当前采用的更新顺序是：

1. `global prior`
2. `slot prior`
3. `slot posterior`
4. `global posterior`

写成公式：

```text
ḡ_t = T_g(g_{t-1}, Pool(S_{t-1}))
r_t^i = Graph(S_{t-1})_i
s̄_t^i = T_s(s_{t-1}^i, r_t^i, ḡ_t)
Ô_t^{s,i} = H_s(s̄_t^i, ḡ_t)
O_t^g = E_g(x_t)
O_t^s = E_s(x_t; S̄_t, ḡ_t)
A_t = M(S̄_t, Ô_t^s, O_t^s, O_t^g)
S_t = U_s(S̄_t, Ô_t^s, O_t^s, A_t, O_t^g)
g_t = U_g(ḡ_t, O_t^g, Pool(S_t))
```

含义：

- `global prior` 先给出场景级转移条件
- `slot prior` 再在该条件下预测对象状态
- `slot posterior` 使用当前帧观测先修正对象状态
- `global posterior` 最后吸收当前图像观测和更新后的 slot summary

### 5.4 Predicted Observation

动力学不直接把下一时刻图像作为第一目标，而是先预测“本帧应该观测到什么”。

定义：

```text
Ô_t^{s,i} = (p̂_obs_t^i, q̂_obs_t^i, ĉ_obs_t^i)
```

建议：

- `p̂_obs`：预测的相对位置观测
- `q̂_obs`：预测的视觉码 belief
- `ĉ_obs`：预测的观测质量

这样做的目的：

- 让状态预测和观测校正之间有显式桥梁
- 在无监督视频建模中更自然地定义训练目标
- 后续支持无图像条件下的 rollout / imagination

### 5.5 Matching 与生命周期管理

#### 5.5.1 Matching

当前建议：

- 训练时使用带 `dustbin` 的 soft matching
- 推理如果需要离散轨迹，可选 Hungarian / LAPJV 做硬化

不建议直接把普通 Gumbel-Softmax 作为主匹配器，原因是：

- 它更适合单分类选择
- 不天然保证双向唯一性
- 对 `unmatched / rebirth / missing observation` 支持不够自然

更适合的形式是扩展代价矩阵：

```text
C ∈ R^((K+1)×(M+1))
```

其中最后一行和最后一列分别代表 `dustbin`。

#### 5.5.2 生命周期

第一版不使用硬状态机，而使用连续变量：

- `e_t^i`：存在性
- `u_t^i`：不确定性
- `r_t^i`：可复用程度

直觉：

- 遮挡时不直接把别的对象信息写进旧 slot
- 旧 slot 的 `e` 下降、`u` 上升、`a` 保持 sticky
- 当 slot 变得足够“可复用”后，再允许其吸收未匹配 observation

这比 `ACTIVE / OCCLUDED / INACTIVE` 更适合真实视频和长时序。

### 5.6 当前从简版模块划分

第一版建议只保留下列 7 个核心模块：

- `T_g`：Global Prior
- `Graph`：Relation Graph
- `T_s`：Slot Prior
- `H_s`：Predicted Observation Heads
- `M`：Soft Matcher
- `U_s`：Slot Posterior
- `U_g`：Global Posterior

组件级数据流：

```text
x_t -> GlobalEncoder -> O_t^g
x_t -> StaticSlotEncoder -> O_t^s
g_{t-1}, S_{t-1} -> T_g -> ḡ_t
S_{t-1} -> Graph -> r_t
S_{t-1}, r_t, ḡ_t -> T_s -> S̄_t
S̄_t, ḡ_t -> H_s -> Ô_t^s
S̄_t, Ô_t^s, O_t^s, O_t^g -> M -> A_t
S̄_t, Ô_t^s, O_t^s, A_t, O_t^g -> U_s -> S_t
ḡ_t, O_t^g, Pool(S_t) -> U_g -> g_t
```

### 5.7 与世界模型的关系

本框架的目标不是只做被动视频 tracking，而是为后续带行为交互的世界模型预留状态空间。

当前从简版做法：

- 在接口层面预留 `action` 输入
- 第一版不把 action 路由器和复杂 action-conditioned transition 加入主训练路径
- 先把被动视频下的 belief-state filter 做稳定

未来可以自然扩展为：

```text
ḡ_t = T_g(g_{t-1}, Pool(S_{t-1}), a_{t-1})
s̄_t^i = T_s(s_{t-1}^i, r_t^i, ḡ_t, a_{t-1})
```

也就是说：

- 现在先固定状态与更新接口
- 之后再把 action 注入 prior transition

### 5.8 第一版训练目标（最小集合）

当前建议先保留最小自监督目标：

- `L_p`：predicted position observation 对齐 actual position observation
- `L_q`：predicted code belief 对齐 actual code evidence
- `L_c`：predicted confidence 对齐 actual confidence
- `L_u`：uncertainty calibration / consistency

若需要再增加一项，优先增加：

- `L_post`：posterior consistency 的轻量正则

当前不建议第一版就引入：

- 复杂像素重建主目标
- 过多全局辅助头
- 复杂生命周期分类损失
- 过早的 action-conditioned 训练路径

### 5.9 当前结论

截至当前讨论，推荐优先实现的版本是：

- `slot` 作为主状态
- `global` 作为紧凑条件状态
- `observation` 与 `state` 明确分离
- 使用 predicted observation 连接先验预测与观测校正
- 使用带 `dustbin` 的 soft matching 和连续生命周期变量
- 先完成被动视频下的最小可行动力学系统，再向 action-conditioned world model 扩展

### 5.10 Action 注入接口（为后续世界模型预留）

当前结论：

- 需要现在就预留 action 注入接口
- 但第一版不把 action 分支加入主训练路径
- action 主要注入 prior transition，不作为 posterior correction 的主输入

#### 5.10.1 统一 action token

建议把真实 action 或未来 latent action 统一编码为：

```text
u_{t-1} = E_a(a_{t-1})
```

其中：

- `a_{t-1}`：动作输入
- `u_{t-1}`：统一 action token

若当前没有动作标注，也建议在接口上保留 `u_{t-1}`，训练时可默认置零。

#### 5.10.2 Global Prior 中的 action 注入

action 第一条通路进入 `Global Prior`：

```text
ḡ_t = T_g(g_{t-1}, Pool(S_{t-1}), u_{t-1})
```

理由：

- 动作可能影响相机、ego-state 或场景级上下文
- global 分支需要感知动作引起的整体条件变化
- 这也与 Dreamer / RSSM 的“action 进入 prior”原则一致

同时从 `ḡ_t` 和 `u_{t-1}` 中读出一个广播动作条件：

```text
c_t^{act} = W_c[ḡ_t, u_{t-1}]
```

该向量广播给所有 slot。

#### 5.10.3 ActionRouter：动作到 slot 的软绑定

不建议把 action 平均作用到所有 slot。建议增加一个轻量的 `ActionRouter`：

```text
ω_t^i = softmax_i( K(s_{t-1}^i)^T Q[ḡ_t, u_{t-1}] )
u_{t-1}^{dir,i} = ω_t^i · W_v u_{t-1}
```

含义：

- `ω_t^i`：第 `i` 个 slot 接收该动作的强度
- `u_{t-1}^{dir,i}`：路由到该 slot 的动作信号

当前建议：

- 第一版使用 soft routing，不使用 hard routing
- `ω_t^i` 先作为标量门，而不是向量门
- 真实动作作用于少数 slot，再由关系图传播间接影响

#### 5.10.4 Slot Prior 中的 action 注入

slot 先验更新写成：

```text
r_t^i = Graph(S_{t-1})_i
s̄_t^i = T_s(s_{t-1}^i, r_t^i, ḡ_t, c_t^{act}, u_{t-1}^{dir,i})
```

当前建议 action 主要作用于：

- `p_t^i`：强作用
- `e_t^i`：中等作用
- `h_t^i`：中等到强作用

当前不建议第一版让 action 强作用于：

- `a_t^i`：appearance / code belief
- `u_t^i`：uncertainty

原因：

- 动作首先改变运动与交互，而不是默认改变外观 identity
- uncertainty 更适合由 posterior mismatch 驱动，而不是由 action 直接驱动

#### 5.10.5 Posterior 中不直接注入 action

当前建议保持：

```text
S_t = U_s(S̄_t, Ô_t^s, O_t^s, A_t, O_t^g)
g_t = U_g(ḡ_t, O_t^g, Pool(S_t))
```

理由：

- prior 负责 action-conditioned transition
- posterior 负责 observation-conditioned correction
- 这种因果分工与 Dreamer / RSSM 更一致，也更容易稳定训练

#### 5.10.6 当前从简版实现建议

如果当前阶段仍以无动作视频建模为主，建议：

- 保留 action 接口
- 默认 `u_{t-1} = 0`
- 先不启用 action loss 和 action router 训练
- 先把 passive belief-state filter 训练稳定

未来扩展时，可以自然替换为：

- 真实 action token
- latent action token
- language / goal token 的统一条件接口

这保证当前框架不会与后续世界模型断开。
