# 复现、闭包与身份边界

## 文件来源

`metadata/source_manifest.json`记录原工程相对路径、导出相对路径、原文件SHA256及处理方式。核心vendor、kaggle_b5、train/study/local768/long/cloud训练脚本均按原始字节复制；原工程和同步sources只读，不在本次导出中移动、修改或清理。

共享 `src/vdcp/train.py` 仅从原文件AST源片段提取 `seed_everything`、`worker_init`、`rng_state`，函数体不变，去掉所有其他模型构造器及训练入口依赖。data→inference、evaluate→data/inference/regions、study→calibrate/improvements、local768→train/study、long/cloud→local768动态闭包均保留。vendor中通用EfficientNet规格表属于上游共享代码，B5入口只实例化efficientnet-b5，不保留其他方案模型工厂。

新 `b5_archive.py`、infer/evaluate/prepare_b5_data/continue_b5_archive.py 为归档入口，输入路径可显式指定。推理直接取材于原个人证据 `复评与测速.py` 的B5分支，去掉同时冻结B3的流程；阈值沿用float64数组比较及 `>=`，避免阈值边界误差。没有复制包含其他模型的冻结选择文件。

## 历史与新训练

初始416训练、72轮后768训练及云5090训练源码保留用于可追溯。`cloud_b5_compute.py`和`cloud_b5_validation.py`动态加载local768并验证 `d061e49271210ede89233139f00a85d7588f5c7be52bee510f88157a9ca64979`；local768继续验证train/study原始字节。所有这些校验保留。

历史源身份是遍历整个src及trainer产生的，因此精简副本的源身份必然不同，历史续训入口会如实拒绝把此目录声明为原冻结源。历史云部署还需原inputs_manifest、迁移preflight及完整状态。这些不是推理依赖，不能伪造身份或删除检查来“跑通”。

`continue_b5_archive.py`为新的谱系：显式校验父checkpoint SHA256；要求完整model/optimizer/scheduler/scaler/RNG/config/epoch；调用冻结engine组件和整组梯度；保留BN运行缓冲、训练affine、768裁剪、有效batch18及尾组归一化；每轮用本项目统一FP32评估并保存last与epoch快照。micro=1调用本地两遍引擎，micro=3/6/9/18调用云端计算引擎。新源及划分哈希、父身份写入lineage，不重写旧身份。完整模型训练/中断恢复未执行过验收；pending_validation检查明确拒绝未经完成验证的父状态。

初始训练默认从固定预训练encoder文件开始，需自行取得 `pretrained/efficientnet-b5-b6417697.pth`，SHA256 `b641769706a245cea68d309c3d35a864b6317e686e315fe79ed3d81e78c6f51c`。数据放在仓库 `data/raw/`，不进入Git。运行 `python scripts/train_kaggle_b5.py --allow-local --device cuda --stop-at 60`；此历史入口有实际batch18/16显存预检，不能用于本机4GB显卡生产训练。模型不需要预训练文件即可加载正式checkpoint推理。

## 数据生成

数据从Kaggle获取；归档不重新分发比赛图片。RLE按原 `rle.py` 的列优先布局解码，四类别位掩码打包。历史ZIP解压、尺寸/标签审计、SHA256分组及迭代多标签划分过程保留于prepare_data/preparation/splits。实际使用现有CSV，不重新随机切分；历史种子20261008与20261009。原数据和留出集审计及评价说明在`docs/source_evidence`。该目录的原审计脚本只作为历史来源，含原路径时不作为新入口。

## 测试取舍

保留B5结构、精度、整组梯度/RNG、BN、尾组、长续训状态、云计算/runtime、统一评价、RLE、区域、划分、准备测试。一个依赖外部72轮完整权重的历史集成测试显式skip，未把模型放入普通Git。watchdog测试将150ms启动阈值扩大至2秒、子进程超时10秒至60秒，并按比例延长进度测试心跳；原因是Windows Python启动超过原阈值，生产runtime源码未改。原始首次测试失败及复测记录保留在仓库外审计目录。

不声称重新训练200轮；历史72—200指标是既有证据。内部测试并非从未使用的独立盲测。完整末轮本地模型导出只有model/epoch/pending_validation/export_provenance，不能替代完整200轮续训状态。

## Kaggle私有训练包

`python scripts/build_kaggle_b5_bundle.py --weights pretrained/efficientnet-b5-b6417697.pth --output work/vdcp_b5_private.zip` 可按已有train/val清单构建私有输入包；不加入内部test图片。该builder在副本中将旧其他方案目录的locked_protocol来源改为固定`reference/locked_protocol.json`并带上group evaluator。`kaggle/b5_fpn_t4.ipynb`保持原件用于初始训练，未在Kaggle执行验收。包包含比赛数据，仅按数据条款用于自己的私有环境，不上传公开仓库。

## 本轮修复与验证补充

安装包内保留`vdcp/_reference/group_unified_v2.py`固定副本；仓库评价器存在时优先使用仓库原件，否则读取包内资源，两条路径均执行相同SHA256检查。wheel已实际构建、安装到独立target，并在仓库外工作目录及Python隔离模式下成功导入及评价。vendor LICENSE和upstream_manifest也随wheel打包。

新训练谱系除source/parent/CSV之外，还对train+val每张真实图片及PNG mask分别计算SHA256，排序后的资产表摘要绑定到lineage，完整资产表写`data_assets.json`。准备入口拒绝重复(image,class)注释，并逐类别将解码后的非空状态与归档CSV的c1..c4核对；正样本缺失注释会失败，比赛表中省略的纯负样本可以按清单重建。补充测试覆盖这两种缺失状态与重复注释。

来源清单的`source_sha256`严格为原工程文件摘要；`export_sha256`为导出后文件摘要。新文件只有后者，清单本身不自包含摘要。历史实验记录中的原D盘路径仅为来源说明，不被新推理/评估入口读取。

中断恢复限制：若验证期间中断，`pending_validation=True` 的 last.pt 不能由新入口直接续接；应保留该文件作诊断，使用上一个验证完成的 epoch 快照（首轮则使用原始父checkpoint）在新的空输出目录重跑该轮。不要删除原始父模型。
