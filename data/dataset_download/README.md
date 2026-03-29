# Dataset Download

这个目录只放数据集下载、解压、整理脚本。

当前数据目录约定：

- 原始压缩包和解压数据统一放到 `data/dataset/`
- 可训练 / 推理的数据集 loader 放到 `data/dataloader/`

当前下载脚本：

- `download_clevr.py`
- `download_coco.py`
- `download_davis.py`
- `download_youtube_vos.py`
- `download_imagenet.py`
- `download_movi_c.py`
- `download_cater.py`
- `download_opnet.py`
- `download_ovis.py`
- `download_lvos_v2.py`

示例：

```bash
python3 data/dataset_download/download_movi_c.py   --root data/dataset/movi_c   --include-segmentations
```

```bash
python3 data/dataset_download/download_cater.py   --root data/dataset/cater
```

```bash
python3 data/dataset_download/download_opnet.py   --root data/dataset/opnet
```

```bash
python3 data/dataset_download/download_ovis.py   --root data/dataset/ovis   --images-url <ovis-images-archive-url>   --annotations-url <ovis-annotations-archive-url>
```

```bash
python3 data/dataset_download/download_lvos_v2.py   --root data/dataset/lvos_v2
```

说明：

- `download_movi_c.py` 通过官方 TFDS `gs://kubric-public/tfds` 物化 `MOVi-C` 到本地帧目录
- `download_cater.py` 默认使用官方公开 Box 链接
- `download_opnet.py` 默认使用 OPNet 官方项目页当前公开的 SharePoint 链接；如果返回 HTML 错误页，说明上游分享链接已失效或需要刷新
- `download_lvos_v2.py` 默认使用官方 Google Drive 共享链接，需要可用的 `gdown`
- `download_ovis.py` 需要显式传入 URL；截至 2026-03-29，OVIS 官方页面仍主要指向比赛页/百度网盘入口，没有稳定公开直链
- 训练与推理阶段请从 `data.dataloader` 导入对应 loader
- `ImageNet` 通常需要你先从官方渠道手动获得压缩包
- `DAVIS` 和 `YouTube-VOS` 可能存在版本和访问限制，建议传入明确的数据源 URL
