# tools/ 工具脚本说明

> 全部脚本通过**项目内虚拟环境**运行，不要直接双击 `.py` 文件。
> 每次运行会自动归档到 `runs/<时间戳>-<任务>/`（README.md 是人读摘要，见 docs/01 §6）。

| 脚本 | 用途 | 运行命令 |
|---|---|---|
| `download_subset.py` | 下载数据集子集：dry-run 预检体积、5GB 熔断、断点续传、自动解压 | `.venv\Scripts\python tools\download_subset.py --dry-run`<br>确认后：`.venv\Scripts\python tools\download_subset.py --yes` |
| `extract_frames.py` | 视频抽帧（5fps、近重复剔除、模糊过滤）；图像型数据集只做计数 | `.venv\Scripts\python tools\extract_frames.py` |
| `convert_to_yolo_pose.py` | 标注转 YOLO-pose 格式，自动划分 train/val，生成 `sop_pose.yaml` | `.venv\Scripts\python tools\convert_to_yolo_pose.py` |
| `check_dataset.py` | 数据质检：统计 + 12 张可视化抽查图（输出到运行归档 samples/） |
| `prelabel_hands.py` | **路线 D**：MediaPipe 预标注（左/右手两类 × 21 关键点）→ 时间块 7/2/1 划分 → `datasets/converted/sop_hands/`（含 split.json 固化） | `.venv\Scripts\python tools\prelabel_hands.py` |
| `compare_models.py` | 两版手部模型在同一 test 集评估并输出差异报告（`--a runs/xxx --b runs/yyy`） | `.venv\Scripts\python tools\compare_models.py --a v1run --b v2run` |
| `make_demo_video.py` | 从训练集生成回放演示视频（给 run 模式用，无摄像头时） | `.venv\Scripts\python tools\make_demo_video.py` |
| `archive_runs.py` | 轮转 `runs/`（保留最近 5 次），移入 `archive/` 并做一次 `archive:` git 提交 | `.venv\Scripts\python tools\archive_runs.py` |

## 标准流程（M1 已全部跑通）

```
download_subset.py --yes  →  extract_frames.py  →  convert_to_yolo_pose.py
→  check_dataset.py  →  src/main.py train  →  src/main.py run --video xxx.mp4
```

## 配置

- 数据源与子集：`config/dataset.yaml`（`active_source` 切换 `coco_val2017` / `assembly101`）
- SOP 步骤与 ROI：`config/sop_demo.yaml`
- Assembly101 为 gated 数据集：先在 huggingface.co 授权 `cvml-nus/assembly101`，
  再 `set HF_TOKEN=<你的读token>`，然后 `download_subset.py --dry-run` 查看将下载内容
