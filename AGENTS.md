# Repository Guidelines

## 项目结构与模块组织
- `ViTP_IN/`: ViTP 预训练主模块（InternVL），核心代码在 `internvl/`，配置在 `ViTP_configs/`，脚本在 `tools/`。
- `mmrotate/`、`mmseg/`、`opencd/`: 下游任务子项目（旋转检测/语义分割/变化检测），结构遵循 OpenMMLab。
- `mmcv/`: 自定义 MMCV 依赖。
- `Figs/`: 文档插图；顶层 `README.md` 与 `ViTP 项目复现指南.md` 为使用入口。

## 构建、测试与开发命令
- 预训练环境（ViTP_IN）：
  - `cd ViTP_IN`
  - `conda create -n ViTP python=3.9 && conda activate ViTP`
  - `pip install -r requirements.txt`
  - 训练示例：`GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh ViTP_configs/InternVL_1b_remote_sensing_ViTP.sh`
- 目标检测（mmrotate）：
  - `cd mmrotate && pip install -r requirements.txt && pip install -e .`
  - 训练：`sh tools/dist_train.sh ViTP_configs/vitp_dotav2_orcnn.py 8`
  - 测试：`sh tools/dist_test.sh ViTP_configs/vitp_dotav2_orcnn.py work_dirs/vitp_dotav2_orcnn/latest.pth 8`
- 语义分割（mmseg）：`sh tools/dist_train.sh ViTP_configs/vitp_isaid_upernet.py 8`
- 变化检测（opencd）：`sh tools/dist_train.sh ViTP_configs/vitp_s2looking_upernet.py 8`

## 编码风格与命名约定
- Python，4 空格缩进；遵循 OpenMMLab 风格。
- 格式化与导入排序配置见 `mmcv/setup.cfg`、`mmseg/setup.cfg`、`mmrotate/setup.cfg`、`opencd/setup.cfg`（yapf/isort）。
- 配置文件命名：`vitp_<dataset>_<model>.py`（如 `vitp_dotav2_orcnn.py`）。
- 语言规范：代码注释统一使用中文，注释需简洁明确，避免无意义注释。

## 测试指南
- 各子项目使用 `pytest`（如 `cd mmcv && pytest`）。
- 多数用例依赖 CUDA，可使用 `-k` 过滤或跳过 GPU 相关测试。

## 提交与合并请求规范
- 提交信息保持简短直接（例如 `Update README.md`、`update eval`）。
- PR 描述需说明：影响模块、配置、数据集与硬件；涉及训练/评估改动请附指标或日志；文档更新可附截图。

## 数据与权重管理
- 不提交数据集或大体积权重；使用外部存储并在配置中引用路径（参考 `ViTP_IN/ViTP_configs/`）。
