import streamlit as st
import pandas as pd
import altair as alt
import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()

from config import MODELS, USD_TO_JPY, get_api_key, ModelConfig
from providers import LLMResponse, OpenAIClient, AnthropicClient, GoogleClient, XAIClient

st.set_page_config(page_title="LLM性能比較", page_icon="🤖", layout="wide")

CLIENTS = {"openai": OpenAIClient, "anthropic": AnthropicClient, "google": GoogleClient, "xai": XAIClient}

def get_model_params() -> dict:
    """サイドバーからパラメータを取得"""
    return {
        # GPT-5.1
        "gpt51_reasoning": st.session_state.get("gpt51_reasoning", "medium"),
        "gpt51_verbosity": st.session_state.get("gpt51_verbosity", "medium"),
        "gpt51_max_tokens": st.session_state.get("gpt51_max_tokens", 10000),
        # GPT-5 / mini / nano
        "gpt5_reasoning": st.session_state.get("gpt5_reasoning", "medium"),
        "gpt5_verbosity": st.session_state.get("gpt5_verbosity", "medium"),
        "gpt5_max_tokens": st.session_state.get("gpt5_max_tokens", 10000),
        # Claude
        "claude_thinking": st.session_state.get("claude_thinking", False),
        "claude_budget": st.session_state.get("claude_budget", 8000),
        "claude_temp": st.session_state.get("claude_temp", 0.0),
        "claude_max_tokens": st.session_state.get("claude_max_tokens", 10000),
        # Gemini 3 Pro
        "gemini3_thinking_level": st.session_state.get("gemini3_thinking_level", "low"),
        "gemini3_max_tokens": st.session_state.get("gemini3_max_tokens", 10000),
        # Gemini 2.5系（共通）
        "gemini_temp": st.session_state.get("gemini_temp", 0.0),
        "gemini_max_tokens": st.session_state.get("gemini_max_tokens", 10000),
        # Grok（全モデル共通）
        "grok_temp": st.session_state.get("grok_temp", 0.0),
        "grok_max_tokens": st.session_state.get("grok_max_tokens", 10000),
    }

def run_generation(model: ModelConfig, prompt: str, params: dict) -> LLMResponse:
    api_key = get_api_key(model.provider)
    if not api_key:
        return LLMResponse("", 0, 0, 0, model.id, f"APIキー未設定: {model.provider.upper()}_API_KEY", 0)

    client = CLIENTS[model.provider](api_key)

    if model.provider == "openai":
        if "gpt-5.1" in model.id:
            return client.generate(prompt, model.id,
                reasoning_effort=params["gpt51_reasoning"],
                verbosity=params["gpt51_verbosity"],
                max_completion_tokens=params["gpt51_max_tokens"])
        else:
            return client.generate(prompt, model.id,
                reasoning_effort=params["gpt5_reasoning"],
                verbosity=params["gpt5_verbosity"],
                max_completion_tokens=params["gpt5_max_tokens"])
    elif model.provider == "anthropic":
        return client.generate(prompt, model.id,
            extended_thinking=params["claude_thinking"],
            budget_tokens=params["claude_budget"],
            temperature=params["claude_temp"],
            max_tokens=params["claude_max_tokens"])
    elif model.provider == "google":
        if "gemini-3" in model.id:
            return client.generate(prompt, model.id,
                thinking_level=params["gemini3_thinking_level"],
                max_tokens=params["gemini3_max_tokens"])
        else:
            return client.generate(prompt, model.id,
                temperature=params["gemini_temp"],
                max_tokens=params["gemini_max_tokens"])
    else:  # xai（全モデル共通）
        return client.generate(prompt, model.id,
            temperature=params["grok_temp"],
            max_tokens=params["grok_max_tokens"])

def extract_pdf_text(file) -> str:
    """PDFファイルからテキストを抽出"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def get_prompt_input(key: str) -> str:
    prompt = st.text_area("プロンプト", height=200, key=f"{key}_prompt", placeholder="プロンプトを入力...\n\n{file_content} でファイル内容を挿入可能")
    with st.expander("📁 ファイル添付"):
        file = st.file_uploader("ファイル", type=["txt", "md", "csv", "json", "py", "pdf"], key=f"{key}_file")
        if file:
            try:
                if file.name.endswith(".pdf"):
                    content = extract_pdf_text(file)
                    st.success(f"✅ PDF読み込み完了: {file.name} ({len(content):,} 文字)")
                else:
                    content = file.read().decode("utf-8")
                    st.success(f"✅ ファイル読み込み完了: {file.name} ({len(content):,} 文字)")

                st.code(content[:1000])
                if len(content) > 1000:
                    st.caption(f"... 以下省略（残り {len(content) - 1000:,} 文字）")

                if "{file_content}" in prompt:
                    prompt = prompt.replace("{file_content}", content)
                else:
                    prompt = f"{prompt}\n\n{content}" if prompt.strip() else content
            except Exception as e:
                st.error(f"ファイル読み込みエラー: {e}")
    return prompt

def render_sidebar():
    """サイドバーにパラメータ設定UIを表示"""
    with st.sidebar:
        st.header("⚙️ パラメータ設定")

        # GPT-5.1
        st.subheader("GPT-5.1")
        st.selectbox("reasoning_effort", options=["none", "low", "medium", "high"], index=2, key="gpt51_reasoning",
            help="推論の深さ。noneで推論なし")
        st.selectbox("verbosity", options=["low", "medium", "high"], index=1, key="gpt51_verbosity",
            help="出力の詳細度")
        st.slider("max_completion_tokens", 1000, 16000, 10000, 1000, key="gpt51_max_tokens")

        st.divider()

        # GPT-5 / mini / nano
        st.subheader("GPT-5 / mini / nano")
        st.selectbox("reasoning_effort", options=["low", "medium", "high"], index=1, key="gpt5_reasoning",
            help="推論の深さ")
        st.selectbox("verbosity", options=["low", "medium", "high"], index=1, key="gpt5_verbosity",
            help="出力の詳細度")
        st.slider("max_completion_tokens", 1000, 16000, 10000, 1000, key="gpt5_max_tokens")

        st.divider()

        # Claude
        st.subheader("Claude Sonnet / Haiku")
        thinking_on = st.toggle("extended_thinking", key="claude_thinking",
            help="拡張思考モードを有効化")
        if thinking_on:
            st.slider("budget_tokens", 1024, 32000, 8000, 1024, key="claude_budget",
                help="思考トークンの予算")
        else:
            st.slider("temperature", 0.0, 1.0, 0.0, 0.1, key="claude_temp")
        st.slider("max_tokens", 1000, 16000, 10000, 1000, key="claude_max_tokens")

        st.divider()

        # Gemini 3 Pro
        st.subheader("Gemini 3 Pro")
        st.selectbox("thinking_level", options=["low", "high"], index=0, key="gemini3_thinking_level",
            help="推論の深さ。lowで高速、highで精度重視")
        st.slider("max_tokens", 1000, 16000, 10000, 1000, key="gemini3_max_tokens")

        st.divider()

        # Gemini 2.5系
        st.subheader("Gemini 2.5系")
        st.slider("temperature", 0.0, 1.0, 0.0, 0.1, key="gemini_temp")
        st.slider("max_tokens", 1000, 16000, 10000, 1000, key="gemini_max_tokens")

        st.divider()

        # Grok（全モデル共通）
        st.subheader("Grok")
        st.slider("temperature", 0.0, 1.0, 0.0, 0.1, key="grok_temp")
        st.slider("max_tokens", 1000, 16000, 10000, 1000, key="grok_max_tokens")

def main():
    render_sidebar()
    params = get_model_params()

    st.title("LLM性能比較")

    tab1, tab2 = st.tabs(["単体テスト", "比較テスト"])
    all_models = [m for ms in MODELS.values() for m in ms]

    with tab1:
        options = {m.name: m for m in all_models}
        model = options[st.selectbox("モデル", list(options.keys()))]
        prompt = get_prompt_input("single")

        if st.button("実行", type="primary", key="run1"):
            if prompt.strip():
                with st.spinner(f"{model.name} 生成中..."):
                    r = run_generation(model, prompt, params)
                if r.error:
                    st.error(r.error)
                else:
                    st.metric("時間", f"{r.latency_ms/1000:.2f}秒")
                    st.metric("コスト", f"¥{r.calculate_cost(model.input_price, model.output_price) * USD_TO_JPY:.4f}")
                    st.text(r.content)

    with tab2:
        cols = st.columns(4)
        selected = []
        for i, (p, ms) in enumerate(MODELS.items()):
            with cols[i]:
                for m in ms:
                    if st.checkbox(m.name, value=False, key=f"cmp_{m.id}"):
                        selected.append(m)

        prompt = get_prompt_input("compare")

        if st.button("比較実行", type="primary", key="run2"):
            if selected and prompt.strip():
                results = {}
                progress_bar = st.progress(0, text="準備中...")

                for i, m in enumerate(selected):
                    progress_bar.progress(i / len(selected), text=f"{m.name} 生成中... ({i+1}/{len(selected)})")
                    results[m.id] = run_generation(m, prompt, params)

                progress_bar.progress(1.0, text="完了")

                # グラフ・表用データ
                chart_data = []
                for m in selected:
                    r = results[m.id]
                    if not r.error:
                        row = {
                            "モデル": m.name,
                            "時間(秒)": r.latency_ms / 1000,
                            "入力トークン": r.input_tokens,
                            "出力トークン": r.output_tokens,
                            "コスト(¥)": r.calculate_cost(m.input_price, m.output_price) * USD_TO_JPY,
                        }
                        if r.reasoning_tokens > 0:
                            row["思考トークン"] = r.reasoning_tokens
                        chart_data.append(row)
                    else:
                        st.error(f"{m.name}: {r.error}")

                if chart_data:
                    df = pd.DataFrame(chart_data)

                    # グラフ表示
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("⏱️ レスポンス時間")
                        df["時間ラベル"] = df["時間(秒)"].apply(lambda x: f"{x:.2f}秒")
                        bars = alt.Chart(df).mark_bar().encode(
                            x=alt.X("モデル:N", sort=None, title=None),
                            y=alt.Y("時間(秒):Q", title="秒"),
                            color=alt.Color("モデル:N", legend=None),
                        )
                        text = bars.mark_text(dy=-10, fontSize=14).encode(text="時間ラベル:N")
                        st.altair_chart(bars + text, use_container_width=True)

                    with col2:
                        st.subheader("💰 コスト")
                        df["コストラベル"] = df["コスト(¥)"].apply(lambda x: f"¥{x:.4f}")
                        bars = alt.Chart(df).mark_bar().encode(
                            x=alt.X("モデル:N", sort=None, title=None),
                            y=alt.Y("コスト(¥):Q", title="円"),
                            color=alt.Color("モデル:N", legend=None),
                        )
                        text = bars.mark_text(dy=-10, fontSize=14).encode(text="コストラベル:N")
                        st.altair_chart(bars + text, use_container_width=True)

                    # 表
                    st.dataframe(df.drop(columns=["時間ラベル", "コストラベル"]).style.format({"時間(秒)": "{:.2f}", "コスト(¥)": "¥{:.4f}"}), use_container_width=True)

                # レスポンス表示
                st.subheader("📝 レスポンス")
                for m in selected:
                    r = results[m.id]
                    with st.expander(m.name, expanded=True):
                        if r.error:
                            st.error(r.error)
                        else:
                            st.text(r.content)

if __name__ == "__main__":
    main()
