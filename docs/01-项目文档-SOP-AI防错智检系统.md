# 项目文档-SOP-AI防错智检系统（一期：手部动作姿态 · 本地训推一体 Demo）

| 项 | 内容 |
|---|---|
| 文档编号 | 01 |
| 文档状态 | 草稿待评审 |
| 关联文档 | [00-文档制作规范](./00-文档制作规范.md) · [02-数据集方案](./02-数据集方案.md) |

---

## 1. 项目概述

对标 tyenjunai.com 的「AI-SOP 防错智检系统」，以自研 + 开源（Ultralytics **YOLO11**）方式复刻其**软件端核心流程**。

一期定位：**本地单机「训推一体」Demo**——在同一套工程里完成「数据下载转换 → YOLO11 姿态模型训练 → 视频回放实时推理 → 动作判定报警 → 数据落盘归档 → 数据大屏」完整闭环，不依赖服务器与真实摄像头。

## 2. 范围与分期

### 2.1 一期（本项目，当前）

| 模块 | 内容 |
|---|---|
| 动作姿态检测 | **重点：手部动作**。YOLO11-pose 关键点检测 + 规则化动作判定 |
| SOP 管理 | SOP 步骤存 YAML；**v1 即在工作站界面展示** SOP 步骤与当前步骤高亮 |
| 推理输入 | 视频文件回放（当前无真实摄像头） |
| 报警 | 软件级：蜂鸣音 + 界面警示 + 事件落盘 |
| 数据落盘 | 每次运行**自动存档全流程产生的、人能看懂的数据**（见 §6） |
| 数据大屏 | 本地网页，读取本地运行数据做可视化（一期任务） |
| 训推一体 | 同一工程两种运行模式：`train`（训练版 Developer）/ `run`（检测版 Runtime） |

### 2.2 二期（暂不做，仅占位）

零件/状态检测（漏装、错装）、SOP 顺序状态机校验、真实工业相机接入、声光报警器/PLC 硬件联动、核心数据服务器与多工位。

## 3. 已确认决策

| # | 决策点 | 结论 | 确认来源 |
|---|---|---|---|
| 1 | 修改留痕 | git 管理，文档内不写变更记录；过时内容直接删除 | 用户 2026-10-25 |
| 2 | 一期任务 | 动作姿态检测（重点手部动作）+ 数据大屏 | 用户 2026-10-25 |
| 3 | 硬件 | 本机为开发机，GPU 为 **RTX 3050 4GB**，训练与推理同机 | 用户 2026-10-25 |
| 4 | 数据 | 只用公开数据集，**无法自采样本** | 用户 2026-10-25 |
| 5 | 摄像头 | 暂无真实摄像头，用视频文件回放 | 用户 2026-10-25 |
| 6 | 服务器 | 不做，只做本地核心流程 demo | 用户 2026-10-25 |
| 7 | SOP | YAML 文件存储，v1 需在界面展示 | 用户 2026-10-25 |
| 8 | 文档形态 | Markdown（当前即可） | 用户 2026-10-25 |
| 9 | 环境隔离 | 特殊依赖安装在项目内 `.venv/` | 用户 2026-10-25 |
| 10 | 工作站/大屏 UI | **Vue 3**（Vite 构建，大屏用 ECharts），后端 FastAPI 提供接口与推理帧流 | 用户 2026-10-25 |
| 11 | 主数据集 | **Assembly101 子集**，严禁全量下载，只下选定装配任务的子集 | 用户 2026-10-25 |
| 12 | 辅助数据集 | 暂不做（相关章节已删除） | 用户 2026-10-25 |
| 13 | v1 关键点路线 | **C 组合**：先人体 17 关键点预训练直推跑通闭环，M2 后半用 Assembly101 手部标注专训 21 关节精化模型 | 用户 2026-10-25 |
| 14 | git 远程 | 公开仓库（GitHub） | 用户 2026-10-25 |
| 15 | 验收基线 | mAP@50 ≥ 0.70、Runtime ≥ 15 fps（未提异议，默认认可） | 用户 2026-10-25 |
| 16 | 手部模型数据路线 | **路线 D（模型预标注）**：MediaPipe 教师预标注 → v1 训练 → X-AnyLabeling/CVAT 人工修正 → v2 二次训练 → 同测试集对比两版差距；7/2/1 时间块划分；数据不足继续取段 | 用户 2026-10-25（切路已记 agents.log） |

## 4. 系统架构（单机）

```
┌───────────────────────────── 本机（RTX 3050 4GB）─────────────────────────────┐
│                                                                              │
│  datasets/raw ──▶ tools: download/convert ──▶ datasets/converted (YOLO格式)  │
│                                                        │                     │
│                                              ┌─────────▼──────────┐          │
│        src/main.py train                     │  YOLO11-pose 训练   │          │
│        （Developer 模式）──────────────────▶ │  (3050 4GB, AMP)   │          │
│                                              └─────────┬──────────┘          │
│                                                        │ best.pt             │
│  config/sop_demo.yaml ──▶ ┌────────────────────────────▼───────────────────┐ │
│  （SOP 步骤+判定规则）     │  src/main.py run （Runtime 模式）               │ │
│                           │  视频文件回放 → 逐帧姿态推理 → SOP 动作规则判定   │ │
│                           │  → 报警(声音/界面) → 事件与截图落盘              │ │
│                           └────────────────────┬───────────────────────────┘ │
│                                                │ 全程自动归档到 runs/<时间戳>/ │
│                           ┌────────────────────▼───────────────────────────┐ │
│                           │  数据大屏（Vue 3 本地网页，读 runs/ 与事件文件）  │ │
│                           └────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 5. 一期算法方案

### 5.1 模型与判定

| 层 | 方案 |
|---|---|
| 关键点检测 | 人体：`yolo11n-pose.pt`（COCO 17 关键点）微调；手部：**路线 D**——MediaPipe Hands 教师预标注（每手 21 关键点，左/右手两类）蒸馏 YOLO11-pose，两阶段训练对比见 02 §1 |
| 动作判定 | **规则引擎**（非深度模型）：关键点与 ROI/工件区域的相对位置、双手距离、手部运动轨迹 + 连续 N 帧时序平滑 |
| SOP 校验（一期展示级） | 每个步骤绑定一条关键点规则；步骤超时/回退产生告警事件。完整顺序状态机属二期 |
| 部署 | 训练产物 `best.pt` 直接本机推理；可选导出 ONNX |

### 5.2 RTX 3050 4GB 训练参数约束

```bash
yolo pose train data=datasets/converted/sop_pose/sop_pose.yaml \
  model=yolo11n-pose.pt epochs=150 imgsz=640 batch=4 patience=30 workers=2
```

- 显存 4GB：用 `yolo11n-pose`、`imgsz=640`、`batch=4`（OOM 则降 `imgsz=480`）、AMP 混合精度默认开启；
- 数据集子集规模控制见 02 §2（目标 2000–4000 张标注帧）。

### 5.3 SOP YAML（config/sop_demo.yaml，v1 界面展示此文件内容）

```yaml
sop_id: demo_assembly_01
product: 拼装玩具车（数据集子集）
steps:
  - id: 1
    name: 取底盘
    description: 双手从料区拿起底盘放至装配区
    expect:
      rule: hands_in_roi          # 关键点规则名（src 内规则引擎实现）
      roi: assembly_area          # ROI 定义在同文件底部
      stable_frames: 15
  - id: 2
    name: 安装车轮 x4
    description: 依次将四个车轮装入底盘
    expect: { rule: hand_near_object, object: wheel_tray, stable_frames: 15 }
alarm:
  step_timeout_sec: 60            # 步骤超时 → 疑似漏执行告警
  out_of_order: warn              # 步骤回退 → 顺序异常提示
roi:
  assembly_area: { x1: 0.30, y1: 0.35, x2: 0.70, y2: 0.85 }
  wheel_tray:    { x1: 0.05, y1: 0.50, x2: 0.25, y2: 0.90 }
```

## 6. 运行自动归档规范（每次运行强制）

**每一次 train / run / convert / check / download / extract / demo_video 执行，结束后自动在 `runs/YYYYMMDD-HHMMSS-<task>/` 写入：**

| 文件 | 内容（人能看懂） |
|---|---|
| `README.md` | 本次运行摘要：何时、什么任务、输入数据、完整命令、关键结果结论（自动生成，中文） |
| `manifest.json` | 机器可读的参数与输入/输出清单 |
| `log.txt` | 全量控制台输出 |
| `results.csv` | 训练任务：逐轮指标（loss、mAP，ultralytics 原生格式，大屏 `/api/train_metrics` 读取） |
| `events.jsonl` | 回放任务：检测统计与 SOP 事件流（step_done / step_timeout / out_of_order，大屏 events 面板读取） |
| `samples/` | 关键可视化：标注抽查帧、推理叠加帧（annotated.mp4）、报警事件截图 |

**自动归档规则**：`tools/archive_runs.py` 按任务类型轮转（train / run / check / convert / download / extract / demo_video 各自保留最近 5 次），过时运行移入 `archive/runs/`，并自动执行 `git commit -m "archive: ..."`（提交仅含 `archive/ARCHIVE_LOG.md` 清单行）。归档动作 = 一次 git 提交。训练结束时会同时复制 `best.pt` 到 `weights/best.pt`（部署权重，不受轮转影响，run 模式优先取用）。

## 7. 环境与依赖（项目内隔离）

```bash
cd "本项目根目录"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`：`ultralytics`、`torch`（CUDA 版，按 3050 装 cu121 官方轮子）、`opencv-python`、`fastapi`、`uvicorn`、`pyyaml`、`pandas`、`matplotlib`。

CUDA 版 torch 须先于 requirements 安装：`python -m pip install torch --index-url https://download.pytorch.org/whl/cu121`。

**一键启动**：双击根目录 `start_dev.bat`（纯 ASCII，防编码错误）：自动停掉占用 8000/5173 端口的进程 → 首次运行自动创建 `.venv` 并装依赖、`npm install` → 分窗口启动后端 `uvicorn src.api:app`（:8000）与前端 Vite（:5173）。后端 API 入口 `src/api.py`，训练/推理 CLI 入口 `src/main.py`，前端在 `web/`（M2 落地）。

## 8. 里程碑

| 阶段 | 内容 | 退出条件 |
|---|---|---|
| M0 | git 仓库 + 目录骨架 + `.venv` + 数据集选定 | 本文档评审通过、02 §2 主数据集确定 |
| M1 | 下载/抽帧/转换脚本，姿态训练跑通，自动归档验证 | `runs/` 出现含 README.md 的训练归档 |
| M2 | Runtime：视频回放推理 + SOP 规则判定 + 报警 + 落盘 | 演示视频上步骤识别与告警正常 |
| M3 | 数据大屏（Vue 3 前端 + FastAPI 后端，读本地数据） | 浏览器可看实时回放检测状态与统计 |
| M4 | 训推一体整合演示 + 文档定稿 | 全流程一键演示通过 |

## 9. 风险

| 风险 | 对策 |
|---|---|
| 4GB 显存训练 OOM | yolo11n + imgsz 640 + batch 4 + AMP；仍不足降 imgsz 480 或缩小子集 |
| 公开数据集体量大（Assembly101 全量数 TB） | 严禁全量下载：下载脚本按「装配任务 × 相机视角」粒度取子集，dry-run 核对体积后再下（见 02 §2） |
| 公开数据与真实产线视角/光照有差距 | 一期定位即公开数据 demo，不做产线验收；差距记录在 runs/README.md 供二期评估 |
| Ultralytics AGPL-3.0 许可 | 演示/内部无碍；商用前评估商业许可或替换推理实现 |

## 10. 待定项

当前无待定项：v1 关键点路线（C 组合）、git 远程（公开仓库）、验收基线均已确认，见 §3 决策表第 13–15 条。
