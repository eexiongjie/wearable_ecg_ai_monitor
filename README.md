# Wearable ECG AI Monitor

一个面向**医疗电子 / 可穿戴心电监测设备**的完整工程 Demo：支持单导联 ECG 信号仿真、滤波、R 峰检测、心率与 HRV 特征提取、心律异常初筛、模型训练、FastAPI 服务接口和简易 Web 页面。

> 重要说明：本项目仅用于工程学习、课程设计、作品集或毕业设计 Demo，不构成医疗诊断工具，不能替代医生判断或临床医疗器械注册认证。

## 1. 项目功能

- ECG 数据仿真：生成正常、心动过速、心动过缓、节律不齐、混合异常等单导联心电信号。
- 信号预处理：基线漂移抑制、带通滤波、50/60 Hz 工频陷波。
- R 峰检测：自适应阈值 + 峰值间距约束。
- 特征提取：平均心率、最大/最小心率、RR 间期、SDNN、RMSSD、pNN50、信号能量等。
- 异常筛查：基于规则的心动过速、心动过缓、节律不齐、信号质量异常识别。
- AI 模型训练：使用合成数据训练轻量级 RandomForest 窗口分类器。
- API 服务：提供 `/health`、`/simulate`、`/analyze`、`/analyze-csv` 接口。
- Web 页面：支持上传 CSV 并查看异常筛查结果。
- 工程化配置：pytest 单元测试、Dockerfile、GitHub Actions CI。

## 2. 项目结构

```text
wearable-ecg-ai-monitor/
├── src/ecg_ai_monitor/
│   ├── api/                # FastAPI 服务
│   ├── data/               # ECG 仿真器
│   ├── dsp/                # 滤波、R峰检测、特征提取
│   ├── ml/                 # 模型训练与推理
│   ├── screening/          # 异常初筛规则引擎
│   ├── utils/              # IO 与报告工具
│   └── cli.py              # 命令行入口
├── data/sample_ecg.csv     # 示例 ECG 数据
├── models/                 # 模型输出目录
├── reports/                # 分析报告输出目录
├── web/index.html          # 简易前端页面
├── tests/                  # 单元测试
├── scripts/                # 辅助脚本
└── .github/workflows/ci.yml
```

## 3. 快速运行

### 3.1 创建环境

```bash
cd wearable-ecg-ai-monitor
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 3.2 生成示例 ECG 数据

```bash
python -m ecg_ai_monitor.cli simulate --duration 60 --fs 250 --scenario mixed --out data/sample_ecg.csv
```

### 3.3 分析 CSV 数据

```bash
python -m ecg_ai_monitor.cli analyze --input data/sample_ecg.csv --fs 250 --out reports/sample_report.json
```

输出内容包括：信号质量、R 峰数量、平均心率、HRV 指标、异常片段列表和综合结论。

### 3.4 训练轻量 AI 分类模型

```bash
python -m ecg_ai_monitor.cli train --out models/ecg_window_classifier.joblib --n-samples 120
```

### 3.5 启动 API 和 Web 页面

```bash
python -m ecg_ai_monitor.cli serve --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

API 文档：

```text
http://127.0.0.1:8000/docs
```

## 4. CSV 数据格式

最少需要两列：

```csv
time,ecg
0.000,0.012
0.004,0.018
0.008,0.025
```

其中：

- `time`：单位为秒。
- `ecg`：单导联 ECG 采样值，单位可理解为 mV 或归一化幅值。

## 5. 运行测试

```bash
pytest -q
```

## 6. Docker 运行

```bash
docker build -t wearable-ecg-ai-monitor .
docker run -p 8000:8000 wearable-ecg-ai-monitor
```

## 7. 推荐 Git 分支规划

你可以按下面方式开多个分支，方便 GitHub 上展示工程过程：

```bash
git init
git add .
git commit -m "init: wearable ECG AI monitor"
git branch -M main

git checkout -b feature/signal-processing
# 提交 dsp/ 相关代码

git checkout main
git checkout -b feature/ai-screening-api
# 提交 screening/、api/ 相关代码

git checkout main
git checkout -b feature/web-dashboard
# 提交 web/ 相关代码
```

也可以直接使用 `scripts/create_branches.sh` 中给出的命令参考。

## 8. 工程亮点描述

本项目实现了一个面向可穿戴医疗电子设备的单导联心电智能监测系统，围绕 ECG 信号采集后的软件处理流程，完成了信号预处理、R 峰检测、心率与 HRV 特征提取、心律异常初筛、轻量级模型训练和远程接口服务。系统能够对长时间心电数据进行自动分析，输出疑似心动过速、心动过缓和节律不齐片段，并生成结构化报告，可作为医疗电子设备算法验证、远程监护原型系统和嵌入式智能筛查方案的工程基础。
