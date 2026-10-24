# 第三方来源

Historical FPN / EfficientNet 实现取自 https://github.com/khornlund/segmentation-models-pytorch ，固定commit `df7d87f39adffa4b7264d88bd224596af45dc4c1`。

原LICENSE、逐文件upstream及patched SHA256、命名空间和 `np.float -> float` 补丁记录保留于 `src/vdcp/vendor/kh_fpn/LICENSE` 与 `upstream_manifest.json`。此次导出未修改vendor文件。不得用pip安装的新版SMP替代。

统一评价器由组内提供，保留原文件及SHA256 `784722c2d57667a681e61b5637bbe68acd3532ca8bc158a924e2b58eb9f70189`，原始署名与注释未改；其公开再分发许可未另行授予，此归档按用户授权的私有研究仓库使用。

其余项目代码与实验记录来自用户原工程2026-10-24实际磁盘版本，不从旧Git HEAD替代。用户代码和成果的许可权由原作者保留；本整理不为整个项目新增统一开源许可证。

比赛数据不随仓库分发，获取与使用遵守Kaggle及赛事原条款。
