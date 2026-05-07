#!/usr/bin/env bash
set -e

git init
git add .
git commit -m "init: wearable ECG AI monitor full project"
git branch -M main

git checkout -b feature/signal-processing
git checkout main

git checkout -b feature/ai-screening-api
git checkout main

git checkout -b feature/web-dashboard
git checkout main

echo "Created branch skeletons: main, feature/signal-processing, feature/ai-screening-api, feature/web-dashboard"
