#!/bin/bash

echo "📥 讀取 Colab 錯誤 log..."
cat debug_log.txt

echo "🧠 使用 Codex CLI 嘗試修正 main.py..."
codex "請根據以下錯誤訊息修正 main.py：" < debug_log.txt > main_fixed.py

if [ -f main_fixed.py ]; then
    mv main_fixed.py main.py
    echo "✅ 修正成功，已更新 main.py"

    git add main.py
    git commit -m "fix: auto fix by Codex"
    git push origin main
else
    echo "❌ Codex 未成功產生修正內容"
fi
