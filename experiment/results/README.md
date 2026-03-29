# Results

统一实验结果以数据集为一级目录：

```text
experiment/results/
  clevr/
    <run_name>/
      meta/
      ckpt/
      logs/
      curves/
      outputs/
        validation/
        evaluation/
```

单次运行包的内容约定：

- `meta/args.json`：启动参数
- `meta/config.json`：模型、动态模块、数据集配置
- `meta/runtime.json`：设备与运行环境信息
- `meta/summary.json`：本次运行摘要、最佳指标、产物位置
- `ckpt/last.pt`：最后一次状态
- `ckpt/best.pt`：按验证损失选择的最佳状态
- `ckpt/step_*.pt`：按步数周期保存的中间状态
- `logs/train.jsonl`：训练日志
- `logs/validation.jsonl`：验证日志
- `logs/evaluation.jsonl`：评估日志
- `curves/*.png`：训练/验证曲线
- `outputs/validation/step_xxxxxxx/*.png`：与验证同步保存的可视化对比
- `outputs/evaluation/*/*.png`：最终评估可视化输出
