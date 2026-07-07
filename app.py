# -*- coding: utf-8 -*-
"""
竹溪社问答助手 · 网页版（Streamlit）
本地预览：  streamlit run app.py
部署后玩家打开链接即可提问。

"""

import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ===== 服务商配置（智谱 GLM）=====
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"   # OpenAI 用户改成 None
CHAT_MODEL = "glm-4-flash"
EMBED_MODEL = "embedding-3"


def get_api_key():
    k = os.environ.get("LLM_API_KEY", "")
    if k:
        return k
    try:
        return st.secrets.get("LLM_API_KEY", "")
    except Exception:
        return ""


@st.cache_resource(show_spinner="正在准备知识库…")
def build_chain(api_key):
    docs = TextLoader("knowledge.txt", encoding="utf-8").load()
    # chunk 放大 + overlap 加大：让每段更完整（尤其问答对不被拆散）
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    ).split_documents(docs)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=api_key, base_url=BASE_URL)
    store = FAISS.from_documents(chunks, embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 5})  # 多检索几段，回答更全
    # max_tokens 控制成本；temperature=0 保证准确不乱编
    llm = ChatOpenAI(model=CHAT_MODEL, api_key=api_key, base_url=BASE_URL,
                     temperature=0, max_tokens=500)
    prompt = ChatPromptTemplate.from_template(
        "你是竹溪社的官方问答小助手。请用亲切、热情、口语化的语气回答，"
        "像社团里热心的学长学姐，让人感到温暖友好；可以适当用“～”和一个小 emoji，但别太多。"
        "回答要具体、有条理，把【资料】里相关的信息尽量讲清楚，不要敷衍或只说一句话。\n"
        "请只根据下面的【资料】回答；如果资料里确实没有，就友好地说"
        "“这个我暂时没有资料，可以直接问社团工作人员哦～”，绝不编造。"
        "与竹溪社无关的问题（闲聊、写代码、常识等）请礼貌婉拒并引导回社团话题。\n\n"
        "【资料】\n{context}\n\n【问题】{question}\n\n【回答】"
    )
    def fmt(ds):
        return "\n\n".join(d.page_content for d in ds)
    return (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


st.set_page_config(page_title="竹溪社问答助手", page_icon="🎋")
st.title("🎋 竹溪社问答助手")
st.caption("关于竹溪社的任何问题都可以问我～（怎么加入 / 有哪些活动 / 一个人来会不会尴尬…）")

api_key = get_api_key()
if not api_key:
    st.error("未配置 API key。请在部署平台的 Secrets（或本地环境变量 LLM_API_KEY）里设置。")
    st.stop()

chain = build_chain(api_key)

# —— 简单防滥用：单次会话提问上限 + 单条长度限制（防烧穿）——
MAX_Q_PER_SESSION = 20   # 每次会话最多问 20 次，超过需刷新
MAX_Q_LEN = 200          # 单个问题最多 200 字

# 推荐问题：帮不熟悉社团的玩家快速上手（在这里增减即可）
SUGGESTIONS = [
    "怎么加入竹溪社？",
    "剧本社交是什么？",
    "有哪些活动？",
    "一个人来会不会尴尬？",
    "参加需要花钱吗？",
    "不是留学生可以参加吗？",
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "count" not in st.session_state:
    st.session_state.count = 0

# 推荐问题：始终显示在顶部，方便随时点其他问题
st.markdown("💡 **不知道问什么？点这些试试：**")
cols = st.columns(2)
for i, s in enumerate(SUGGESTIONS):
    if cols[i % 2].button(s, key=f"sug_{i}", use_container_width=True):
        st.session_state.pending = s
        st.rerun()
st.divider()

# 展示历史消息
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# 问题来源：点击的推荐问题 或 手动输入
typed = st.chat_input("输入你的问题…")
q = st.session_state.pop("pending", None) or typed

if q:
    if st.session_state.count >= MAX_Q_PER_SESSION:
        st.warning("本次对话的提问次数已达上限，刷新页面即可继续～")
    elif len(q) > MAX_Q_LEN:
        st.warning(f"问题有点长，请精简到 {MAX_Q_LEN} 字以内～")
    else:
        st.session_state.count += 1
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("思考中…"):
            ans = chain.invoke(q)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()
