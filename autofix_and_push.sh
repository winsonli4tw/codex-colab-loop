#!/usr/bin/env bash
# 自動修正 + Git commit/push
set -e

echo "🧠 GPT-4o 自動修正中..."
python3 autofix.py

echo "🚀 Git push..."
git add main.py
git commit -m "fix: auto-patch via GPT-4o"
git push
echo "✅ 完成！已推送修正版本到 GitHub"
