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
- `download_lpwm.py`

示例：

```bash
python3 data/dataset_download/download_movi_c.py   --root data/dataset/movi_c   --include-segmentations
```

```bash
python3 data/dataset_download/download_cater.py   --root data/dataset/cater
```

```bash
python3 data/dataset_download/download_coco.py   --root data/dataset/coco   --include-test
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

LPWM 数据集入口：

```bash
python3 data/dataset_download/download_lpwm.py --list
```

```bash
python3 data/dataset_download/download_lpwm.py --dataset sketchy
```

```bash
python3 data/dataset_download/download_lpwm.py --all-auto
```

说明：

- `download_lpwm.py` 默认把数据放到 `data/dataset/<dataset>`
- `sketchy`、`bair`、`bridge`、`panda`、`ogbench`、`mario`、`obj3d128` 支持自动下载到本地
- `shapes` 不需要下载，LPWM 会在线生成
- `langtable` 需要基于 upstream 的 `datasets/langtable_preparation.py` 自行预处理
- `balls`、`phyre` 只有 MEGA 链接，脚本只会提示你手动提供归档或自行下载后再挂载
- `traffic` 在 upstream 中有 loader，但这里没有稳定公开下载源
- `balls_occlusion`、`langtable_action`、`obj3d128_img`、`sketchy_action` 都是数据别名，会映射到它们的主数据集目录

说明：

- `download_movi_c.py` 通过官方 TFDS `gs://kubric-public/tfds` 物化 `MOVi-C` 到本地帧目录
- `download_cater.py` 默认使用官方公开 Box 链接
- `download_opnet.py` 默认使用 OPNet 官方项目页当前公开的 SharePoint 链接；如果返回 HTML 错误页，说明上游分享链接已失效或需要刷新
- `download_coco.py` 默认下载 COCO 2017 的 `train2017`、`val2017`、`annotations_trainval2017`，加上 `--include-test` 会再下载 `test2017` 和 `image_info_test2017`
- `download_lvos_v2.py` 默认使用官方 Google Drive 共享链接，需要可用的 `gdown`
- `download_ovis.py` 需要显式传入 URL；截至 2026-03-29，OVIS 官方页面仍主要指向比赛页/百度网盘入口，没有稳定公开直链
- 训练与推理阶段请从 `data.dataloader` 导入对应 loader
- `ImageNet` 通常需要你先从官方渠道手动获得压缩包
- `DAVIS` 和 `YouTube-VOS` 可能存在版本和访问限制，建议传入明确的数据源 URL
