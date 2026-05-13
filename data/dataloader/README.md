# Dataloader

这个目录放置训练、推理、评估可直接使用的数据集 loader。

当前支持：

- `clevr.py`：静态图像 loader，返回 `images`
- `clevr_cogent.py`：静态图像 loader，返回 `images`
- `coco.py`：静态图像 loader，返回 `images`，并在存在 `instances_*.json` / `image_info_*.json` 时附带 COCO 标注元信息
- `imagenet.py`：静态图像 loader，返回 `images`
- `davis.py`：视频序列 loader，返回 `frames`
- `youtube_vos.py`：视频序列 loader，返回 `frames`

与 pipeline 的对齐方式：

- `script/static.py` 期望 batch 中存在 `images: [B, C, H, W]`
- `script/dynamic.py` 期望 batch 中存在 `frames: [B, T, C, H, W]`

统一实验入口 `experiment/scripts/run_experiment.py` 会通过 `--dataset` 自动选择这里对应的 loader，并通过以下参数统一下发数据配置：

- `--data-root`
- `--train-split`
- `--eval-split`
- `--image-size`
- `--max-items`
- `--frames-per-clip`
- `--frame-stride`
- `--normalize`

默认数据目录使用仓库下的 `data/dataset/`。
