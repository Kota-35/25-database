# 25-database

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Polars](https://img.shields.io/badge/polars-1.37+-orange.svg)](https://pola.rs/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)



## Get Started

### 1. プロジェクトのクローン

```bash
git clone https://github.com/Kota-35/25-database.git
cd 25-database
```

### 2. miseのインストール

```bash
# macOS/Linux
curl https://mise.run | sh

# または Homebrew
brew install mise
```

詳しくはこちら (https://mise.jdx.dev/getting-started.html) を参照してください。

### 3. 開発環境のセットアップ

```bash
# Python と uv のインストール
mise install

# 依存パッケージのインストール
uv sync
```

### 4. 分析の実行

```bash
# 分析1の実行
uv run src/analysis_01.py

# 分析2の実行
uv run src/analysis_02.py

# 分析3の実行
uv run src/analysis_03.py
```

実行結果の図は `figures/` ディレクトリに保存されます。

## Project Structure

```
25-database/
├── src/           # 分析スクリプト
├── figures/       # 生成された図
├── latex/         # レポート
└── 25-database-class/  # データファイル
```
