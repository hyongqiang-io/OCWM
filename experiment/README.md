# Experiment

`experiment/` 现在以统一入口为中心组织：

- `scripts/run_experiment.py`：训练与评估共用的主入口，只通过参数切换数据集、模式和超参数
- `scripts/`：兼容包装脚本、冒烟脚本与辅助工具
- `results/`：按数据集划分的实验结果包

推荐入口：

```bash
python3 experiment/scripts/run_experiment.py --mode train --dataset clevr
python3 experiment/scripts/run_experiment.py --mode eval --dataset clevr --checkpoint <ckpt>
```

当前统一入口支持：

- 静态数据集：`clevr`、`clevr_cogent`、`coco`、`imagenet`
- 时序数据集：`davis`、`youtube_vos`
- 统一参数面：`--data-root`、`--train-split`、`--eval-split`、`--image-size`、`--max-items`、`--batch-size`、`--eval-batch-size`、`--epochs`、`--steps`、`--val-every-steps`、`--save-every-steps`、`--frames-per-clip`、`--frame-stride`、`--device`、`--checkpoint`、`--resume`

结果目录约定：

- `experiment/results/<dataset>/<run_name>/meta`
- `experiment/results/<dataset>/<run_name>/ckpt`
- `experiment/results/<dataset>/<run_name>/logs`
- `experiment/results/<dataset>/<run_name>/curves`
- `experiment/results/<dataset>/<run_name>/outputs/validation`
- `experiment/results/<dataset>/<run_name>/outputs/evaluation`
