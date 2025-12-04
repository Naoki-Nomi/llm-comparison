# LLM性能比較ツール

OpenAI / Anthropic / Google / xAI の各モデルを実務タスクで比較検証するStreamlitアプリケーション。

## 機能

- **単体テスト**: 1つのモデルでタスクを実行
- **比較テスト**: 複数モデルで同じタスクを実行して比較

## できること

- 選択したモデルで、自由にプロンプトをカスタマイズしてAPIコール可能
- 複数のモデルで同時に同じプロンプトでAPIコールすることが可能
- 各モデルのパラメータをサイドバーで設定可能
- 各モデルのレスポンス速度と概算の日本円コストを確認可能
- PDFなどのファイルアップロードにも対応

## セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/Naoki-Nomi/llm-comparison
cd llm-comparison
```

### 2. APIキーを設定

```bash
cp .env.example .env
```

`.env` ファイルを編集し、各プロバイダーのAPIキーを設定:

```env
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=AIza-xxx
XAI_API_KEY=xai-xxx
```

### 3. 起動

#### Docker

```bash
docker compose up --build
```

### 4. アクセス

ブラウザで http://localhost:8501 を開く

## ディレクトリ構成

```
llm-comparison/
├── docker-compose.yml
├── Dockerfile
├── .env                 # APIキー（Git管理外）
├── .env.example         # サンプル
├── .gitignore
├── .streamlit/
│   └── config.toml      # Streamlit設定（ホットリロード等）
├── requirements.txt
├── README.md
└── src/
    ├── app.py           # Streamlitメイン
    ├── config.py        # モデル・タスク定義
    └── providers/       # 各APIクライアント
        ├── __init__.py
        ├── base.py
        ├── openai_client.py
        ├── anthropic_client.py
        ├── google_client.py
        └── xai_client.py
```

## 対応モデル

| プロバイダー | モデル | ドキュメント |
|-------------|--------|-------------|
| OpenAI | GPT-5.1, GPT-5, GPT-5 mini, GPT-5 nano | [Models](https://platform.openai.com/docs/models) |
| Anthropic | Claude Sonnet 4.5, Claude Haiku 4.5 | [Models](https://docs.anthropic.com/en/docs/about-claude/models) |
| Google | Gemini 3 Pro, Gemini 2.5 Pro, Gemini 2.5 Flash | [Models](https://ai.google.dev/gemini-api/docs/models/gemini) |
| xAI | Grok 4, Grok 4.1 Fast, Grok 3 Mini | [Models](https://docs.x.ai/docs/models) |

## パラメータ設定

サイドバーで各モデルのパラメータを設定できます。

| モデル | 設定可能なパラメータ |
|--------|---------------------|
| GPT-5.1 | reasoning_effort (none/low/medium/high), verbosity, max_completion_tokens |
| GPT-5 / mini / nano | reasoning_effort (low/medium/high), verbosity, max_completion_tokens |
| Claude Sonnet / Haiku | extended_thinking, budget_tokens, temperature, max_tokens |
| Gemini 3 Pro | thinking_level (low/high), max_tokens ※temperature=1.0推奨のため設定なし |
| Gemini 2.5系 | temperature, max_tokens |
| Grok | temperature, max_tokens |

## 開発

### ホットリロード

Docker環境では `src/` 配下のファイルを編集すると自動でリロードされます。

### モデル追加

`src/config.py` の `MODELS` に追加:

```python
MODELS = {
    "openai": [
        ModelConfig("gpt-5.1", "GPT-5.1", "openai", 1.25, 10.00),
        # 新しいモデルを追加
    ],
    ...
}
```

## ライセンス

MIT
