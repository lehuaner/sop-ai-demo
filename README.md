# SOP-AI Demo

SOP-AI 行为防错智检系统 · 本地训推一体 Demo（一期）。对标 tyenjunai.com 的软件端核心流程。

- 文档：[docs/00-文档制作规范](docs/00-文档制作规范.md) · [01-项目文档](docs/01-项目文档-SOP-AI防错智检系统.md) · [02-数据集方案](docs/02-数据集方案.md)
- 一键启动开发服务：双击 `start_dev.bat`（后端 :8000 / 前端 :5173，自动清端口）
- 训练/推理：`python src/main.py train` · `python src/main.py run --video xxx.mp4`
- 数据流水线：`python tools/download_subset.py --dry-run` → `extract_frames.py` → `convert_to_yolo_pose.py` → `check_dataset.py`
- 归档轮转：`python tools/archive_runs.py`（每次归档 = 一次 `archive:` git 提交）

| 目录 | 内容 |
|---|---|
| `src/` | 主程序（CLI train/run、后端 API、运行归档器） |
| `tools/` | 数据下载/抽帧/转换/质检/归档脚本 |
| `config/` | 数据集与 SOP 配置（YAML） |
| `runs/` | 每次运行自动归档（README + manifest + log + 指标 + 可视化） |
| `archive/` | 过时产物归档（ARCHIVE_LOG.md 为唯一入库留痕） |
| `docs/` | 全部文档 |
| `web/` | Vue 3 前端（M2 落地） |
| `weights/` | 预训练与部署权重（`best.pt` 等，不入 git） |
| `AGENTS.md` / `agents.log` | 代理协作须知与执行偏差记录 |

License: MIT（代码）；数据集遵循各自许可（Assembly101 为 CC BY-NC 4.0，仅限非商用研究演示）。
