# app.py  （B 輕量版：無 GPT-2，純規則＋統計特徵）
from model_logic import split_sentences, sentence_feature_score, highlight_text

import streamlit as st
import re
import os
import statistics
from typing import List, Tuple, Dict

import PyPDF2
import docx
import pandas as pd
import altair as alt

# ==============================
# 1. Streamlit 基本設定 & CSS
# ==============================

st.set_page_config(
    page_title="AI Content Detector (Lite)",
    page_icon="🤖",
    layout="wide"
)

# 自訂樣式：動態漸層背景 + 置中
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        color: #ffffff;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    h1, h2, h3 {
        text-align: center;
    }

    .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a1a1a !important;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1.05rem;
    }

    .stFileUploader {
        padding: 16px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(5px);
        margin-bottom: 16px;
    }

    .stButton>button {
        background: white !important;
        color: #e73c7e !important;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        padding: 0.4rem 2.2rem;
        transition: transform 0.2s, box-shadow 0.2s;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stButton>button:hover {
        transform: scale(1.04);
        box-shadow: 0 0 12px rgba(255, 255, 255, 0.6);
    }
/* ✅ 隱藏 Streamlit 系統 UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================
# 2. 工具函式：讀檔 & 斷句
# ==============================


def extract_text_from_file(uploaded) -> str:
    """支援 txt / pdf / docx 三種格式"""
    if uploaded is None:
        return ""

    filename = uploaded.name.lower()

    try:
        if filename.endswith(".txt"):
            return uploaded.read().decode("utf-8", errors="ignore")

        if filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text

        if filename.endswith(".docx"):
            doc = docx.Document(uploaded)
            return "\n".join(p.text for p in doc.paragraphs)

    except Exception as e:
        st.error(f"讀取檔案失敗：{e}")
        return ""

    # 其他副檔名一律當作純文字
    try:
        return uploaded.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""




# ==============================
# 3. 規則型偵測邏輯（輕量版）
# ==============================

PUNCT_SET = set(".,;:!?。！？、，；：…")




# ==============================
# 4. UI：標題區
# ==============================

st.markdown(
    "<h1 style='font-size:3.2rem;'>AI Content Detector (Lite)</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='font-size:1.1rem; text-align:center; opacity:0.9;'>"
    "貼上文字或上傳檔案，讓輕量版偵測器幫你估算內容看起來像 AI 還是 Human。"
    "<br>此版本不載入大型模型，適合課堂 Demo 與線上展示。</p>",
    unsafe_allow_html=True,
)

# ==============================
# 5. 中央容器：選項 + 輸入
# ==============================

if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""

def on_file_upload():
    uploaded = st.session_state.uploaded_file_key
    text = extract_text_from_file(uploaded)
    if text:
        st.session_state["user_text"] = text

left, center, right = st.columns([1, 6, 1])

with center:
    # 語言選擇（其實只影響說明文字，規則本身語言無關）
    col_opt, col_tag = st.columns([3, 1])
    with col_opt:
        lang = st.selectbox(
            "選擇語言 / Select Language",
            ["Traditional Chinese (中文)", "English / Mixed"],
            index=0
        )
    with col_tag:
        st.markdown(
            "<div style='margin-top:30px; text-align:center; "
            "background:rgba(0,0,0,0.2); padding:8px 10px; "
            "border-radius:999px; font-size:0.8rem;'>"
            "⚡ Rule-based Lite</div>",
            unsafe_allow_html=True,
        )

    st.file_uploader(
        "Upload a text / PDF / Word",
        type=["txt", "pdf", "docx"],
        key="uploaded_file_key",
        on_change=on_file_upload,
    )

    text_input = st.text_area(
        "Paste your text here",
        value=st.session_state["user_text"],
        height=260,
        placeholder="Start writing or paste your text here...",
    )

    st.write("")
    run_btn = st.button("🔍 Start Analysis")

# ==============================
# 6. 分析與結果顯示
# ==============================

if run_btn:
    final_text = text_input.strip()
    with center:
        if not final_text:
            st.warning("⚠️ 請先輸入文字或上傳檔案內容再開始分析。")
        else:
            sentences = split_sentences(final_text)

            if len(sentences) == 0:
                st.warning("⚠️ 內容太短或無法斷句，請再貼多一點文字。")
            else:
                with st.spinner("Analyzing content (rule-based)..."):
                    sentence_probs = []
                    sentence_lens = []
                    chart_rows = []

                    for idx, s in enumerate(sentences, start=1):
                        ai_prob, feats = sentence_feature_score(s)
                        sentence_probs.append(ai_prob)
                        sentence_lens.append(feats["length"])

                        short_s = s[:25] + "…" if len(s) > 25 else s
                        chart_rows.append(
                            {
                                "SentenceID": f"句 {idx}",
                                "Probability": ai_prob,
                                "Length": feats["length"],
                                "Summary": short_s,
                                "Text": s,
                            }
                        )

                    avg_ai = sum(sentence_probs) / len(sentence_probs)
                    if len(sentence_lens) >= 2:
                        mean_len = statistics.mean(sentence_lens)
                        std_len = statistics.pstdev(sentence_lens)
                        burstiness = std_len / mean_len if mean_len > 0 else 0
                    else:
                        burstiness = 0.0

                    # 1) 分數卡片
                    st.markdown(
                        f"""
<div style="background-color: white; padding: 30px; border-radius: 12px; margin-bottom: 22px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
  <table style="width: 100%; border-collapse: collapse;">
    <tr>
      <td style="width: 50%; text-align: center; border-right: 2px solid #eee; padding: 20px;">
        <h3 style="color: #666; margin: 0;">AI Probability</h3>
        <h1 style="color: {'#e73c7e' if avg_ai >= 55 else '#23d5ab'}; 
                   font-size: 2.5em; margin: 15px 0;">
          {avg_ai:.0f}%
        </h1>
        <p style="color: #888; margin: 0;">整體判斷</p>
      </td>
      <td style="width: 50%; text-align: center; padding: 20px;">
        <h3 style="color: #666; margin: 0;">Burstiness Score</h3>
        <h1 style="color: #333; font-size: 2.5em; margin: 15px 0;">
          {burstiness:.2f}
        </h1>
        <p style="color: #888; margin: 0;">句長起伏（越高越人味）</p>
      </td>
    </tr>
  </table>
  
  <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
  
  <div style="text-align: center;">
    <p style="color: #444; font-weight: bold; margin: 0;">
      {'整體看起來偏 AI 風格' if avg_ai >= 55 else '整體看起來偏 Human 風格'}
    </p>
    <p style="color:#777; font-size:0.85rem; margin-top:8px;">
      ※ 此為輕量級 heuristic 模型，僅供 Demo 與學習使用，非正式鑑定工具。
    </p>
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

                    # 2) 詳細句子高亮
                    highlighted_html = highlight_text(sentences, sentence_probs)
                    st.markdown(
                        f"""
<div style="background-color: white; color: #333; padding: 26px;
            border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            margin-bottom: 22px; line-height: 1.9; font-size: 1.05rem;">
  <h3 style="color:#333; margin-bottom:14px; font-weight:bold;
             border-bottom:1px solid #eee; padding-bottom:8px;">
    📝 詳細分析（句子層級）
  </h3>
  <div style="text-align:left;">{highlighted_html}</div>
  <p style="font-size:0.9rem; color:#666; margin-top:16px;">
    <span style="background-color:#fee2e2;color:#991b1b;padding:2px 6px;
                 border-radius:4px;">紅色</span>：高度疑似 AI，
    <span style="background-color:#fef3c7;color:#92400e;padding:2px 6px;
                 border-radius:4px;">黃色</span>：介於 AI / Human 之間，
    <span style="background-color:#dcfce7;color:#166534;padding:2px 6px;
                 border-radius:4px;">綠色</span>：較像 Human 風格。
  </p>
</div>
""",
                        unsafe_allow_html=True,
                    )
                  # 3) Altair 圖表
                    if chart_rows:
                        df_chart = pd.DataFrame(chart_rows)

                        # ✅ 建立 risk 欄位
                        def risk_level(p):
                            if p >= 80:
                                return "High"
                            elif p >= 50:
                                return "Medium"
                            else:
                                return "Low"

                        df_chart["risk"] = df_chart["Probability"].apply(risk_level)

                        # ✅ dynamic_height 一定要在 if 裡面
                        dynamic_height = max(260, len(df_chart) * 38)

                        chart = (
                            alt.Chart(df_chart)
                                .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                                .encode(
                                    x=alt.X(
                                        "Probability:Q",
                                        title="AI 可能性 (%)",
                                        scale=alt.Scale(domain=[0, 100]),
                                    ),
                                    y=alt.Y(
                                        "SentenceID:N",
                                        sort=None,
                                        title="句子索引",
                                    ),
                                    color=alt.Color(
                                        "risk:N",
                                        scale=alt.Scale(
                                            domain=["High", "Medium", "Low"],
                                            range=["#e73c7e", "#facc15", "#23d5ab"]
                                        ),
                                        legend=alt.Legend(
                                            title="風險等級",
                                            titleColor="black",
                                            labelColor="black",)
                                    )
                                )
                                .properties(height=dynamic_height)
                        )

                        chart = (
                            chart
                                .add_selection()   # 你原本的互動保留
                                .encode(
                                    tooltip=["SentenceID", "Probability", "Length", "Text"],
                                )
                                .properties(
                                    height=dynamic_height,
                                    background="rgba(255,255,255,0.9)",
                                    padding={"left": 10, "right": 10, "top": 10, "bottom": 10},
                                    title=alt.TitleParams(
                                        text="Sentence-level AI Probability (Rule-based)",
                                        fontSize=16,
                                        color="black"
                                        ),
                                )
                                .configure_axis(
                                    labelFontSize=11,
                                    titleFontSize=12,
                                    labelColor="#333333",
                                    titleColor="#333333",
                                    grid=False,
                                )
                                .configure_view(strokeWidth=0)
                        )

                        st.altair_chart(chart, use_container_width=True)

                    else:
                        st.info("目前沒有可顯示的資料")
