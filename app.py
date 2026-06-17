import os
import glob
import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #F0F4F8; }
#MainMenu, footer, header { visibility: hidden; }
.header {
    background: linear-gradient(90deg, #0F2240 0%, #1E5F99 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
}
.header h1 { color: white; margin: 0; font-size: 1.5rem; }
.header p { color: #A8CFEE; margin: 4px 0 0; font-size: 0.85rem; }
.bubble-user {
    background: #1E5F99;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    max-width: 75%;
    margin-left: auto;
    margin: 8px 0 8px auto;
    display: block;
    font-size: 0.92rem;
}
.bubble-bot {
    background: white;
    color: #1a1a2e;
    border-radius: 4px 18px 18px 18px;
    padding: 12px 18px;
    max-width: 78%;
    margin: 8px 0;
    border: 1px solid #DDE8F0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    font-size: 0.92rem;
}
.bubble-oos {
    background: #FFFBF0;
    color: #7D5A00;
    border: 1px solid #FDCB6E;
    border-left: 4px solid #FDCB6E;
    border-radius: 4px 18px 18px 18px;
    padding: 12px 18px;
    max-width: 78%;
    margin: 8px 0;
    font-size: 0.92rem;
}
.source-chip {
    display: inline-block;
    background: #E8F4FD;
    color: #0F5486;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    margin-right: 5px;
    margin-top: 6px;
    border: 1px solid #BDD9F0;
}
section[data-testid="stSidebar"] { background: #0F2240 !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li { color: #A8CFEE !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: white !important; }
section[data-testid="stSidebar"] .stButton button {
    background: #1E4D7A;
    color: #E0F0FF;
    border: 1px solid #2E6EA6;
    border-radius: 8px;
    font-size: 0.82rem;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

HR_KEYWORDS = [
    "leave", "salary", "payroll", "ctc", "wfh", "work from home",
    "remote", "hybrid", "performance", "review", "appraisal", "pip",
    "conduct", "policy", "policies", "holiday", "maternity", "paternity",
    "probation", "onboarding", "offboarding", "resign", "resignation",
    "separation", "notice", "travel", "expense", "reimbursement",
    "laptop", "device", "byod", "security", "data", "posh",
    "harassment", "icc", "complaint", "bonus", "increment", "promotion",
    "grade", "benefit", "insurance", "pf", "provident", "gratuity",
    "shift", "attendance", "hr", "employee", "zyro", "handbook",
    "induction", "training", "code of conduct", "sick", "casual",
    "earned", "comp off", "compensatory", "joining", "full and final",
    "fnf", "allowance", "per diem", "working hours", "office", "manager",
]

REFUSAL_MESSAGE = (
    "I am sorry, I can only answer questions related to Zyro Dynamics HR policies. "
    "Your question appears to be outside the scope of the HR policy documents. "
    "Please reach out to the relevant team or resource for assistance."
)

def is_hr_related(question):
    q = question.lower()
    return any(kw in q for kw in HR_KEYWORDS)

@st.cache_resource(show_spinner=False)
def load_pipeline():
    groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    os.environ["GROQ_API_KEY"] = groq_key
    pdf_dir = st.secrets.get("PDF_DIR", os.getenv("PDF_DIR", "./hr_docs"))
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = Path(path).name
        all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.7},
    )
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an HR assistant for Zyro Dynamics Pvt. Ltd.
Your ONLY knowledge source is the HR policy document excerpts provided below.
RULES:
1. Answer ONLY from the context below.
2. Do NOT use outside knowledge.
3. Mention which policy document your answer comes from.
4. Be concise and professional.
5. If not found say: I could not find specific information about this in the HR policy documents.
--- HR POLICY CONTEXT ---
{context}
--- END CONTEXT ---"""),
        ("human", "{question}"),
    ])
    def fmt(docs):
        return "\n\n".join(
            f"[{i}. {doc.metadata.get('source','Unknown')}]\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
        )
    chain = (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever, len(chunks), len(pdf_paths)

# Sidebar
with st.sidebar:
    st.markdown("## 🏢 Zyro Dynamics")
    st.markdown("### HR Help Desk")
    st.markdown("---")
    with st.spinner("Loading HR documents..."):
        try:
            chain, retriever, n_chunks, n_pdfs = load_pipeline()
            pipeline_ok = True
            st.success("✅ Pipeline Ready!")
            st.metric("Policy Documents", n_pdfs)
            st.metric("Indexed Chunks", n_chunks)
        except Exception as e:
            pipeline_ok = False
            st.error(f"Error: {e}")
    st.markdown("---")
    st.markdown("### 💡 Quick Questions")
    quick = [
        "How many earned leaves per year?",
        "What is the WFH policy?",
        "Explain the PIP process",
        "What does POSH policy cover?",
        "Notice period during probation?",
    ]
    for q in quick:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q
    st.markdown("---")
    st.markdown("### 📚 Policy Documents")
    for d in ["Company Profile","Employee Handbook","Leave Policy",
              "WFH Policy","Code of Conduct","Performance Review",
              "Compensation & Benefits","IT & Data Security",
              "POSH Policy","Onboarding & Separation","Travel & Expense"]:
        st.markdown(f"• {d}")

# Header
st.markdown("""
<div class="header">
  <h1>🏢 Zyro Dynamics HR Help Desk</h1>
  <p>Ask anything about company policies — Leave, WFH, Performance, POSH, Travel & more</p>
</div>
""", unsafe_allow_html=True)

# Chat state
if "history" not in st.session_state:
    st.session_state.history = [{
        "role": "bot",
        "content": "👋 Hello! I am your Zyro Dynamics HR assistant. Ask me anything about company policies!",
        "sources": [],
        "oos": False,
    }]

# Render chat
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        if msg.get("oos"):
            st.markdown(f'<div class="bubble-oos">⚠️ {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            chips = "".join(f'<span class="source-chip">📄 {s}</span>' for s in msg.get("sources", []))
            content = msg["content"].replace("\n", "<br>")
            st.markdown(f'<div class="bubble-bot">{content}<br>{chips}</div>', unsafe_allow_html=True)

# Handle prefill
prefill = st.session_state.pop("prefill", None)
user_input = st.chat_input("Ask an HR policy question...")
if not user_input and prefill:
    user_input = prefill

# Process input
if user_input:
    if not pipeline_ok:
        st.error("Pipeline not ready!")
    else:
        st.session_state.history.append({"role": "user", "content": user_input, "sources": [], "oos": False})
        with st.spinner("Searching HR policies..."):
            if not is_hr_related(user_input):
                answer = REFUSAL_MESSAGE
                sources = []
                oos = True
            else:
                docs = retriever.invoke(user_input)
                sources = list({doc.metadata.get("source", "Unknown") for doc in docs})
                answer = chain.invoke(user_input)
                oos = False
        st.session_state.history.append({"role": "bot", "content": answer, "sources": sources, "oos": oos})
        st.rerun()

st.markdown("<br><center style='color:#99AABB; font-size:0.78rem;'>Zyro Dynamics HR Help Desk · Answers based on internal policy documents only</center>", unsafe_allow_html=True)