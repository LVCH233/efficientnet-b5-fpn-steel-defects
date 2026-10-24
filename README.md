# EfficientNet-B5 + Historical FPN

钢板表面缺陷分割归档，仅提供 EfficientNet-B5 + FPN 方案。正式模型选自第135轮；历史训练至200轮。代码保留固定历史版 FPN，不能用新版 segmentation-models-pytorch 替换。

本地整理副本不含模型或比赛图片。GitHub 发布及附件恢复验收由交付记录确认；未发布前没有可用下载链接。原工程保持完整，本项目不执行本地删除。

## 安装

Python 3.12（已验证），Python至少3.11。CPU示例：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python -m pytest -q
```

CUDA 12.1 示例：先以 `--index-url https://download.pytorch.org/whl/cu121` 安装 torch==2.5.1，再安装其余依赖。无需 torchvision、timm 或 segmentation-models-pytorch。Windows命令亦可用 `.venv/Scripts/python.exe`。CPU可推理，训练需要足够显存CUDA设备。

## 模型与数据

正式附件文件名 `best_epoch135.pt`，350865452字节，SHA256：

```text
7e97d666c9164f4c34dfb8ef88cbc120df5862b5d7dab5fd20e510fce230ce56
```

模型须从本仓库正式Release取得（发布后补真实链接）。入口默认核验此哈希。原始checkpoint包含模型、optimizer、scheduler、scaler、RNG和身份信息，因此只加载可信来源；不将350MB模型提交普通Git。

数据从 [Kaggle Severstal](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data) 按比赛条款自行取得。内部test来自原有标注训练集，不是Kaggle无标签test_images。保留 `data/splits/{train,val,test}.csv` 的8797/1885/1886张清单，不重新随机切分。外部数据可按以下方式生成按位压缩PNG掩码：

```powershell
python scripts/prepare_b5_data.py --annotations E:/severstal/train.csv --images E:/severstal/train_images --masks E:/severstal/masks
```

支持 `ImageId_ClassId,EncodedPixels` 和 `ImageId,ClassId,EncodedPixels` 两种注释表。每个PNG第0—3位对应类别1—4；入口检查图片尺寸和划分清单SHA256，已有掩码不同则拒绝覆盖。历史嵌套ZIP准备过程另保留 `scripts/prepare_data.py`；其中split阶段为历史过程说明，复现时直接使用已归档清单。

## 单图推理

```powershell
python scripts/infer_b5.py --checkpoint E:/models/best_epoch135.pt --image E:/severstal/train_images/0025bde0c.jpg --output outputs/example.npz --device cuda
```

无GPU时用 `--device cpu`。NPZ包含float32概率和uint8最终四通道掩码；相邻JSON记录参数和像素数。输入保持1600×256，OpenCV BGR第0通道，`(x0/255-0.3439)/0.0383`。FP32、batch1、Sigmoid、无TTA；阈值`>= [0.75,0.65,0.65,0.85]`，8连通区域最小面积`[256,1024,256,1024]`。

## 统一评估

```powershell
python scripts/evaluate_b5.py --checkpoint E:/models/best_epoch135.pt --images E:/severstal/train_images --masks E:/severstal/masks --split data/splits/val.csv --output outputs/val.json --device cuda
```

`--limit 5`仅做诊断子集，不称为正式验证。固定评价器 `reference/group_unified_v2.py` 及SHA256受代码强制检查，`cov=.3, inside=.5, tol=15, min_area=0`。这里min_area=0不取消预测端面积过滤。

| 集合 | 图像 | G | D | FP | C |
|---|---:|---:|---:|---:|---:|
| 验证 |1885|3029|2507|511|2481|
| 内部测试 |1886|2932|2456|462|2421|

检出率=D/G，误报率=FP/G，准确率=C/(D+FP)。验证为82.77%、16.87%、82.21%。正式选模：FP≤511时优先D，再按准确率、较少FP、较早轮次打破平局。内部测试历史上参与模型比较及错误分析，不能称为完全未使用的盲测集。`results/`保留72—200轮逐轮CSV、protocol和135轮结果；不补造前71轮同口径曲线。

## 训练与续训

详见 [复现与来源](docs/REPRODUCTION.md)。初始入口为原字节保留的 `scripts/train_kaggle_b5.py`，裁剪256×416；768续训的两遍前向、整组Dice、BN缓冲冻结代码保留原件。训练损失为0.6 BCE + 0.4(1-global Dice)，Dice的1e-7只加分母。

精简目录改变了源码身份。历史 `continue_b5_local_768.py`、`continue_b5_local_768_long.py`、`continue_b5_cloud5090.py` 强制校验历史父状态和部署清单，不能把精简目录冒充原训练目录。新入口 `scripts/continue_b5_archive.py` 明确建立新谱系，复用冻结数学计算，不声称重放了历史200轮：

```powershell
python scripts/continue_b5_archive.py --parent E:/models/best_epoch135.pt --parent-sha256 7e97d666c9164f4c34dfb8ef88cbc120df5862b5d7dab5fd20e510fce230ce56 --data-root E:/b5-training-layout --output outputs/new-lineage --target-epoch 136 --micro-batch 1 --device cuda:0
```

外部训练目录应含`data/splits/`清单与`data/raw/train_images/`、`data/raw/masks/`；保留有效批量18及尾组实际归一化。此新入口的完整大模型训练尚未执行。模型型epoch200副本缺少optimizer/RNG，不能据其无缝续训；云端完整末轮未在本地归档。

## 来源与验证

[第三方署名](THIRD_PARTY_NOTICES.md)、[逐文件来源](metadata/source_manifest.json)、[复现边界](docs/REPRODUCTION.md)。上游代码许可证保留在vendor目录；本项目不擅自对用户全部成果套用新许可证。正式单图结果已与历史保存最终掩码逐像素对照；最新测试与完整验证以交付验收记录为准。

2026-10-24独立入口完成1885张完整验证，G=3029、D=2507、FP=511、C=2481，与历史结果一致；见 [本次验证记录](results/fresh_validation_20261024.json)。
