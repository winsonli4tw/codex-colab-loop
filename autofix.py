#!/usr/bin/env python3
"""
autofix.py
----------
讀取 debug_log.txt（Colab 產生的錯誤訊息）與 main.py，
呼叫 OpenAI GPT-4o 自動修正程式，並 **覆寫** main.py。
"""

import os
import pathlib
import textwrap
import openai

# 1️⃣ 先確定環境變數已設定
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("⚠️  請先 export OPENAI_API_KEY=YOUR_KEY")
openai.api_key = api_key

# 2️⃣ 讀取檔案
debug_log = pathlib.Path("debug_log.txt").read_text(encoding="utf-8", errors="ignore")
source_code = pathlib.Path("main.py").read_text(encoding="utf-8", errors="ignore")

# 3️⃣ 建立 prompt
prompt = textwrap.dedent(
    f"""
    以下是 Colab 執行 main.py 時的完整錯誤訊息，請先分析錯誤原因後修正程式碼。
    ----------------[Error Log]----------------
    {debug_log}
    ------------------------------------------

    以下是 main.py 的完整內容。請**覆寫**並輸出修正後的完整程式碼：
    ----------------[main.py]------------------
    ```python
    {source_code}
    ```
    ------------------------------------------

    修正要求：
    1. 必須保留原來功能，僅修正錯誤或潛在例外。
    2. 完整輸出新的 main.py，勿省略任何 import 或程式碼行。
    3. 若需解釋，可用註解寫在程式碼內。不額外輸出說明文字。
    """
)

# 4️⃣ 呼叫 GPT-4o
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
)

fixed_code = response.choices[0].message.content.strip()

# 5️⃣ 去掉 ```python ``` 包裝（若模型加上了）
if fixed_code.startswith("```"):
    fixed_code = fixed_code.split("```", 2)[1]  # 取中間內容
    if fixed_code.lstrip().startswith("python"):
        fixed_code = fixed_code.split("\n", 1)[1]  # 移除 "python" 標籤

# 6️⃣ 覆寫 main.py
pathlib.Path("main.py").write_text(fixed_code, encoding="utf-8")
print("✅ main.py 已用 GPT-4o 修正並覆寫完成")
