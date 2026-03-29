# Script Package

`script/` 用于放置与训练、推理、评估直接相关的 pipeline 代码。

当前包含：

- `static.py`：单帧训练、推理、评估，以及静态损失
- `dynamic.py`：视频训练、推理、评估，以及时序损失
- `common.py`：AMP 与指标转换等共享工具

`module/` 仅保留模型结构、子模块、动态状态单元和配置定义。
