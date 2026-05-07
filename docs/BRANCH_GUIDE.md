# GitHub 分支上传建议

如果你想让 GitHub 上看起来像一个比较完整的工程项目，而不是简单堆代码，可以按以下分支组织：

## main
稳定版完整工程代码，包含 README、接口、测试、示例数据和 CI。

## feature/signal-processing
建议提交内容：

- `src/ecg_ai_monitor/dsp/filters.py`
- `src/ecg_ai_monitor/dsp/rpeaks.py`
- `src/ecg_ai_monitor/dsp/features.py`
- `tests/test_pipeline.py`

提交信息示例：

```bash
git commit -m "feat: add ECG preprocessing and R peak detection"
```

## feature/ai-screening-api
建议提交内容：

- `src/ecg_ai_monitor/screening/engine.py`
- `src/ecg_ai_monitor/ml/`
- `src/ecg_ai_monitor/api/`

提交信息示例：

```bash
git commit -m "feat: add rhythm screening engine and API service"
```

## feature/web-dashboard
建议提交内容：

- `web/index.html`
- README 中 Web 使用说明

提交信息示例：

```bash
git commit -m "feat: add ECG web dashboard"
```

## 最简单上传方式

```bash
git init
git add .
git commit -m "init: wearable ECG AI monitor"
git branch -M main
git remote add origin https://github.com/你的用户名/wearable-ecg-ai-monitor.git
git push -u origin main
```

然后你可以再创建功能分支：

```bash
git checkout -b feature/signal-processing
git push -u origin feature/signal-processing
```
