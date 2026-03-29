# Scripts

推荐直接使用统一入口：

- `run_experiment.py`：训练与评估共用入口

示例：

```bash
python3 experiment/scripts/run_experiment.py --mode train --dataset clevr --image-size 224 --batch-size 4
python3 experiment/scripts/run_experiment.py --mode eval --dataset clevr --checkpoint experiment/results/clevr/<run_name>/ckpt/best.pt
python3 experiment/scripts/run_experiment.py --mode train --dataset davis --frames-per-clip 8 --frame-stride 2
```

当前脚本分为三类：

- `run_experiment.py`：主入口
- `train_clevr_static.py`、`eval_clevr_static.py`：兼容包装脚本，内部会转发到 `run_experiment.py`
- `smoke_train_single_frame.py`、`smoke_train_video.py`、`smoke_infer.py`：链路冒烟验证

这些脚本会调用：

- `script/` 中的 static / dynamic train / eval pipeline
- `data/dataloader/` 中的数据集 loader
- `module/` 中的基础模型结构与 config
