# AGENTS.md — 代理协作须知（Agent Working Agreement）

本文件约束所有在本仓库工作的 AI 代理/自动化流程。人类贡献者同样建议遵守。

## 1. 文档即事实源

| 文档 | 内容 |
|---|---|
| [docs/00-文档制作规范](docs/00-文档制作规范.md) | 命名、结构、git 留痕、目录归属、归档规则（首要） |
| [docs/01-项目文档](docs/01-项目文档-SOP-AI防错智检系统.md) | 范围、决策、架构、里程碑、验收基线 |
| [docs/02-数据集方案](docs/02-数据集方案.md) | 数据源决策与子集控制（5GB 熔断） |
| [tools/README.md](tools/README.md) | 工具脚本用法与标准流程 |

## 2. 硬性规则

- 目录归属：主程序 `src/`、工具 `tools/`、配置 `config/`、前端 `web/`、文档 `docs/`、
  权重 `weights/`、运行产物 `runs/`、归档 `archive/`；**其他位置不得出现脚本或中间产物**；
- git：每次实质变更一笔提交，前缀 `docs: / feat: / fix: / chore: / data: / archive:`；
  **归档动作 = 一次 `archive:` 提交**；大文件（数据/权重/运行产物）不入库；
- 环境隔离：依赖一律装项目内 `.venv`；`.bat` 脚本纯 ASCII；
- 网络：优先国内可达源（ModelScope / hf-mirror / 阿里云 / 清华源）；
  被墙资源（GitHub 直连、HF gated、Google Drive）走本机代理 `127.0.0.1:7897`（env `http_proxy/https_proxy`）；
  **git 推送必须用 `git -c http.sslBackend=openssl -c http.proxy=http://127.0.0.1:7897 push`**（schannel 后端经代理握手会失败）；
  HF gated 数据集 token 在 `.venv/hf_token.txt`（不入库），代理用完即弃，不写入全局配置。

## 3. 偏差记录制度（重要）

**遇到阻碍时：先询问，再决策。** 按文档规范路径实施受阻（授权、网络、数据不可得等）时，
代理必须先向用户说明阻碍与候选方案，**经用户同意后**才能切换路径；未经同意不得自行改道。

**切换路径被用户同意后**，在本目录（项目根）的 `agents.log` 中追加一条记录，格式：

```
[YYYY-MM-DD HH:MM] 偏差：<做了什么与文档不同>
  原因：<为什么>
  涉及文件：<本次偏差改动/新增的文件清单>
  后续：<何时、如何回到文档约定的正轨>
```

只改不记视为违规；`agents.log` 与代码一同提交（`docs:` 前缀）。
等待用户提供的外部输入（token、样品、确认）不算偏差，但应在对话中明确列为阻塞项。
