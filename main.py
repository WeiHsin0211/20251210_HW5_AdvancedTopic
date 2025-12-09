import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import docx
import pandas as pd
import altair as alt
import os
import PyPDF2
# ==========================================
# 1. 設定與風格
# ==========================================
st.set_page_config(page_title="AI Content Detector", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }
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
    h1, h2, h3, p { text-align: center; }
    .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1a1a1a !important;
        border-radius: 12px;
        font-size: 1.1rem;
    }
    .stFileUploader {
        padding: 15px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(5px);
    }
    .stButton>button {
        background: white !important;
        color: #e73c7e !important;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        transition: transform 0.2s;
        display: block;
        margin: 10px auto;
    }
    .stButton>button:hover { transform: scale(1.05); }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 模型載入邏輯 (純淨版)
# ==========================================

@st.cache_resource
def get_model_resource(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None

def compute_perplexity(text: str, tokenizer, model) -> float:
    if not text.strip(): return 0.0
    try:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return float(torch.exp(loss).item())
    except:
        return 0.0

def map_perplexity_to_ai_probability(ppl: float) -> int:
    ppl_clamped = max(5.0, min(100.0, ppl))
    ai_prob = 100 - (ppl_clamped - 5) * (90 / 95)
    return max(5, min(95, int(round(ai_prob))))

def get_highlighted_text(text: str, tokenizer, model) -> tuple[str, float]:
    #sentences = re.split(r'(?<=[.!?。！？])\s*', text)
    split_pattern = r'(?:(?<=[.!?。！？])\s+)|(?:\n+)'
    sentences = re.split(split_pattern, text)
    sentences = [s for s in sentences if s.strip()]
    if not sentences: return "", 0.0
    
    highlighted_parts = []
    total_ai_prob = 0
    valid_count = 0
    
    for sentence in sentences:
        # 👇 修正重點：即使是短字 (標點符號)，也要給它顏色，避免視覺斷裂
        if len(sentence) < 2:
            # 短字元直接視為上一句的屬性，或給予中性綠色
            hl = f'<span style="background-color: transparent; color: black; padding: 2px 4px; border-radius: 4px; margin: 0 2px;">{sentence}</span>'
            highlighted_parts.append(hl)
            continue
        ppl = compute_perplexity(sentence, tokenizer, model)
        ai_prob = map_perplexity_to_ai_probability(ppl)
        total_ai_prob += ai_prob
        valid_count += 1
        
        # 顏色標記邏輯
        if ai_prob > 80:
            hl = f'<span style="background-color: #fee2e2; color: #991b1b; padding: 2px 4px; border-radius: 4px; margin: 0 2px;">{sentence}</span>'
        elif ai_prob > 60:
            hl = f'<span style="background-color: #fef3c7; color: #92400e; padding: 2px 4px; border-radius: 4px; margin: 0 2px;">{sentence}</span>'
        else:
            hl = f'<span style="background-color: #dcfce7; color: #166534; padding: 2px 4px; border-radius: 4px; margin: 0 2px; opacity: 0.8;">{sentence}</span>'
        highlighted_parts.append(hl)
        
    avg_prob = total_ai_prob / valid_count if valid_count > 0 else 0
    return "".join(highlighted_parts), avg_prob

# ==========================================
# 3. UI 介面
# ==========================================

st.markdown("<h1 style='font-size: 3.5rem; text-shadow: 0 4px 10px rgba(0,0,0,0.2);'>AI Content Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2rem; opacity: 0.9; margin-bottom: 20px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>貼上文字或上傳檔案，讓 AI 幫你辨識內容是否由機器生成！</p>", unsafe_allow_html=True)

if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""

def on_file_upload():
    uploaded = st.session_state.uploaded_file_key
    if uploaded is not None:
        try:
            filename = uploaded.name.lower()
            text = ""
            if filename.endswith(".docx"):
                doc = docx.Document(uploaded)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif filename.endswith(".pdf"):
                reader = PyPDF2.PdfReader(uploaded)
                for page in reader.pages:
                    text += page.extract_text() or ""
            else:
                text = uploaded.read().decode("utf-8")
            st.session_state["user_text"] = text
        except Exception as e:
            st.error(f"讀取檔案失敗: {e}")

c1, c2, c3 = st.columns([1, 6, 1]) 

with c2:
    # st.markdown("""
    # <div style="background-color: rgba(255, 255, 255, 0.2); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.3); border-radius: 20px; padding: 15px 25px; margin-bottom: 25px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);">
    #     <h3 style="margin: 0 0 10px 0; color: white; font-size: 1.2rem;">⚙️ 核心設定 (Settings)</h3>
    # </div>
    # """, unsafe_allow_html=True)

    col_opt, col_info = st.columns([3, 1])
    
    with col_opt:
        language_option = st.selectbox(
            "選擇語言模型 / Select Model",
            ["Traditional Chinese (中文)", "English (英文)"],
            index=0
        )
    
    if "Chinese" in language_option:
        # 自動偵測本地資料夾
        if os.path.exists("./model_cn"):
            TARGET_MODEL = "./model_cn"
            status_label = "🟢 中文核心 (Local)"
        else:
            TARGET_MODEL = "uer/gpt2-chinese-cluecorpussmall"
            status_label = "🟠 中文核心 (Online)"
    else:
        TARGET_MODEL = "gpt2"
        status_label = "🔵 English Core"

    with col_info:
        st.markdown(f"""<div style="margin-top: 28px; background: rgba(0,0,0,0.2); color: white; padding: 8px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 0.8rem;">{status_label}</div>""", unsafe_allow_html=True)

    # 載入模型
    with st.spinner(f"正在載入 {status_label}..."):
        tokenizer, model = get_model_resource(TARGET_MODEL)
    
    # 錯誤處理
    if tokenizer is None or model is None:
        if "Chinese" in language_option:
            st.warning("⚠️ 中文模型載入失敗，切換至備援模型 (gpt2)。")
            with st.spinner("切換中..."):
                tokenizer, model = get_model_resource("gpt2")
        else:
            st.error("❌ 無法載入模型。")
            st.stop()

    st.markdown("---")
    st.file_uploader("Upload File (TXT, PDF, DOCX)", type=['txt', 'pdf', 'docx'], key="uploaded_file_key", on_change=on_file_upload)
    
    text_input = st.text_area("Paste text here", value=st.session_state["user_text"], height=250)
    final_text = text_input # 定義 final_text 變數
    
    st.write("")
    detect_button = st.button("🔍 Start Analysis")

# ==========================================
# 4. 分析結果 (絕對修復版：Python 預算顏色)
# ==========================================
if detect_button:
    # 處理變數可能未定義的情況
    if 'final_text' not in locals() or not final_text.strip():
        st.warning("⚠️ 請輸入內容或上傳檔案")
    else:
        with c2:
            with st.spinner("Analyzing content..."):
                # ---------------------------------------------------------
                # 1. 計算邏輯
                # ---------------------------------------------------------
                hl_html, avg_prob = get_highlighted_text(final_text, tokenizer, model)
                
                #sentences = re.split(r'(?<=[.!?。！？])\s*', final_text)
                # ✅ 修改後的寫法 (必須跟上面函式一模一樣)
                split_pattern = r'(?:(?<=[.!?。！？])\s+)|(?:\n+)'
                sentences_list = re.split(split_pattern, final_text)
                lens = [len(s.strip()) for s in sentences_list if len(s.strip()) > 1]
                
                chart_data = []
                for i, s in enumerate(sentences_list):
                    if len(s.strip()) > 1:
                        p = compute_perplexity(s, tokenizer, model)
                        prob = map_perplexity_to_ai_probability(p)
                        
                        short_s = s[:15] + "..." if len(s) > 15 else s
                        
                        # 👇👇👇 修正重點 1：直接在 Python 裡決定顏色，避開 Altair 錯誤 👇👇👇
                        if prob > 80:
                            bar_color = '#e73c7e'  # 紅 (High AI)
                        elif prob > 60:
                            bar_color = '#f59e0b'  # 黃 (Medium)
                        else:
                            bar_color = '#23d5ab'  # 綠 (Human)

                        chart_data.append({
                            "SentenceID": f"句 {i+1}", 
                            "Probability": int(prob), 
                            "Text": s,
                            "Summary": short_s,
                            "BarColor": bar_color  # 把顏色存進去
                        })

                if lens:
                    df_len = pd.DataFrame(lens, columns=['len'])
                    burstiness = df_len['len'].std() / df_len['len'].mean() if df_len['len'].mean() > 0 else 0
                else:
                    burstiness = 0

                # ---------------------------------------------------------
                # 2. 顯示 UI：分數卡片
                # ---------------------------------------------------------
                st.markdown(f"""
<div style="background-color: white; color: black; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 25px;">
<div style="display: flex; justify-content: space-around; align-items: center;">
<div style="flex: 1; border-right: 1px solid #eee; display: flex; flex-direction: column; align-items: center;">
<h3 style="margin: 0; color: #666; font-size: 1rem;">AI Probability</h3>
<h1 style="margin: 5px 0; font-size: 3.5em; color: {'#e73c7e' if avg_prob > 50 else '#23d5ab'};">{avg_prob:.0f}%</h1>
<p style="font-size: 0.9rem; color: #888; margin: 0;">判斷結果</p>
</div>
<div style="flex: 1; display: flex; flex-direction: column; align-items: center;">
<h3 style="margin: 0; color: #666; font-size: 1rem;">Burstiness Score</h3>
<h1 style="margin: 5px 0; font-size: 3.5em; color: #333;">{burstiness:.2f}</h1>
<p style="font-size: 0.9rem; color: #888; margin: 0;">句子節奏變化</p>
</div>
</div>
<div style="text-align: center; margin-top: 10px;">
<p style="color: #444; font-weight: bold; margin: 0;">{'這段文字看起來很像 AI 寫的' if avg_prob > 50 else '這段文字看起來很自然 (Human-written)'}</p>
</div>
</div>
""", unsafe_allow_html=True)

                # ---------------------------------------------------------
                # 3. 顯示 UI：詳細分析報告
                # ---------------------------------------------------------
                st.markdown("### 📝 詳細分析報告")
                st.markdown(f"""
<div style="background-color: white; color: #333; padding: 25px; border-radius: 10px; line-height: 2.0; font-size: 1.05rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
{hl_html}
</div>
""", unsafe_allow_html=True)
                st.caption("🔴 紅色：極高 AI 嫌疑 (>80%) | 🟡 黃色：疑似 AI (60-80%) | 🟢 綠色：人類風格 (<60%)")

                # ---------------------------------------------------------
                # 4. 圖表 (這裡改了！直接讀取 BarColor)
                # ---------------------------------------------------------
                if chart_data:
                    df_chart = pd.DataFrame(chart_data)
                    dynamic_h = max(300, len(chart_data) * 40)
                    
                    c = alt.Chart(df_chart).mark_bar(
                        cornerRadiusTopRight=10,
                        cornerRadiusBottomRight=10
                    ).encode(
                        x=alt.X('Probability', title='AI 可能性 (%)', scale=alt.Scale(domain=[0, 100])),
                        y=alt.Y('SentenceID', sort=None, title='句子索引'),
                        # 👇👇👇 修正重點 2：這裡直接使用我們算好的 BarColor 欄位，不做判斷 👇👇👇
                        color=alt.Color('BarColor', scale=None), 
                        tooltip=['SentenceID', 'Probability', 'Text']
                    ).properties(
                        height=dynamic_h,
                        background='#ffffff'
                    ).configure_axis(
                        labelColor='#333', 
                        titleColor='#333', 
                        grid=False
                    ).configure_view(
                        strokeWidth=0
                    )

                    st.markdown("""<div style="background-color: transparent; border-radius: 12px; padding: 10px; ; margin-top: 20px;"><h4 style="text-align: center; color: white margin: 0 0 15px 0;">📊 句子詳細數據可視化</h4>""", unsafe_allow_html=True)
                    st.altair_chart(c, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)