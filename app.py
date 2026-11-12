
import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Qwen AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. COMPLETE DARK THEME
# ==========================================

st.markdown("""
<style>

/* Main application background */

.stApp {
    background: linear-gradient(
        135deg,
        #080d18 0%,
        #10182b 50%,
        #0b1020 100%
    );
    color: #f8fafc;
}

/* Default text */

p, span, label, li {
    color: #e2e8f0;
}

/* Headings */

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Hide default Streamlit elements */

#MainMenu, footer, header {
    visibility: hidden;
}

/* Main container */

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #0c1424;
    border-right: 1px solid #26344d;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #dbeafe !important;
}

/* Hero header */

.hero {
    background: linear-gradient(
        120deg,
        #172554,
        #312e81,
        #1e3a5f
    );
    border: 1px solid #6366f1;
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 30px;
    box-shadow: 0 0 35px rgba(99, 102, 241, 0.18);
}

.hero h1 {
    color: #ffffff !important;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 10px;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 16px;
}

/* Model status */

.status {
    display: inline-block;
    background: #123b32;
    color: #86efac !important;
    border: 1px solid #22c55e;
    border-radius: 30px;
    padding: 8px 15px;
    font-size: 12px;
    margin-top: 12px;
}

/* Section headings */

.section-title {
    color: #a5b4fc !important;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 18px;
}

/* Chat messages */

[data-testid="stChatMessage"] {
    background: #172235 !important;
    border: 1px solid #34445f !important;
    border-radius: 18px !important;
    padding: 18px !important;
    margin-bottom: 14px;
}

/* Chat message content */

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: #f8fafc !important;
}

/* Prompt input background */

/* Main chat input container */
[data-testid="stChatInput"] {
    background-color: #111c30 !important;
    border: 1px solid #6366f1 !important;
    border-radius: 15px !important;
}

/* Actual input area */
[data-testid="stChatInput"] > div {
    background-color: #111c30 !important;
    border-radius: 15px !important;
}

/* Text area */
[data-testid="stChatInput"] textarea {
    background-color: #111c30 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 16px !important;
    caret-color: #ffffff !important;
}

/* Placeholder */
[data-testid="stChatInput"] textarea::placeholder {
    color: #a5b4c7 !important;
    -webkit-text-fill-color: #a5b4c7 !important;
    opacity: 1 !important;
}

/* Focus effect */
[data-testid="stChatInput"]:focus-within {
    border: 1px solid #818cf8 !important;
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.25);
}

/* Sidebar information cards */

.sidebar-card {
    background: #172235;
    border: 1px solid #34445f;
    border-radius: 14px;
    padding: 17px;
    margin-top: 15px;
}

.sidebar-card p {
    color: #a5b4c7 !important;
    font-size: 12px;
    margin-bottom: 5px;
}

.sidebar-card h4 {
    color: #ffffff !important;
    margin-top: 0;
    font-size: 16px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        90deg,
        #4f46e5,
        #7c3aed
    );
    color: #ffffff !important;
    border: none;
    border-radius: 12px;
    padding: 11px 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #4338ca,
        #6d28d9
    );
    color: #ffffff !important;
    box-shadow: 0 0 18px rgba(139, 92, 246, 0.35);
}

/* Divider */

hr {
    border-color: #334155 !important;
}

/* Spinner */

[data-testid="stSpinner"] {
    color: #c4b5fd !important;
}

/* Empty chat section */

.empty-chat {
    text-align: center;
    padding: 45px 20px;
    background: #111c30;
    border: 1px dashed #475569;
    border-radius: 20px;
    margin-bottom: 25px;
}

.empty-chat h3 {
    color: #ffffff !important;
}

.empty-chat p {
    color: #cbd5e1 !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD QWEN MODEL
# ==========================================

@st.cache_resource
def load_model():

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model_path = "/content/qwen_fresh_cache"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=model_path
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return llm


# ==========================================
# 4. PROMPT TEMPLATE
# ==========================================

prompt = PromptTemplate.from_template(
    """<|im_start|>system
You are a helpful AI assistant.
Answer questions clearly, accurately and simply.
Use conversation history to understand follow-up questions.
<|im_end|>
{history}
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
"""
)


# ==========================================
# 5. INITIALIZE MODEL AND CHAIN
# ==========================================

with st.spinner("Loading Qwen AI model..."):

    llm = load_model()

chain = prompt | llm


# ==========================================
# 6. SESSION STATE
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("### MODEL NAME")
    st.markdown("#### Qwen 2.5 - 0.5B")

    st.markdown("### FRAMEWORK")
    st.markdown("#### LangChain")

    st.markdown("### INTERFACE")
    st.markdown("#### Streamlit")

    st.markdown("### EXECUTION")
    st.markdown("#### GPU Accelerated")

    st.divider()

    st.markdown("### 💬 Conversation")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.caption("Powered by Qwen and LangChain")
    st.caption("Local LLM Project | 2026")

# ==========================================
# 8. HERO SECTION
# ==========================================

st.markdown(
    """
    <div class="hero">
        <h1>✨ Qwen AI Assistant</h1>
        <p>
            Your intelligent conversation partner,
            powered by Qwen2.5 and LangChain.
        </p>
        <div class="status">
            ● MODEL READY &nbsp; | &nbsp; GPU ENABLED
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
# ==========================================
# 9. CHAT SECTION
# ==========================================


if not st.session_state.messages:

    st.markdown(
        """
        <div class="empty-chat">
            <div style="font-size:48px;">💡</div>
            <h3>Start a Conversation</h3>
            <p>
                Ask Qwen anything.
                Your AI assistant is ready!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display previous messages

for message in st.session_state.messages:

    avatar = (
        "🧑‍💻"
        if message["role"] == "user"
        else "🤖"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(message["content"])


# ==========================================
# 10. USER INPUT AND RESPONSE
# ==========================================

user_input = st.chat_input(
    "✨ Type your message here..."
)


if user_input:

    # Prepare conversation history

    history_text = ""

    for message in st.session_state.messages:

        history_text += (
            f"<|im_start|>{message['role']}\n"
            f"{message['content']}"
            f"<|im_end|>\n"
        )

    # Save user message

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message(
        "user",
        avatar="🧑‍💻"
    ):

        st.markdown(user_input)

    # Generate AI response

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner("✨ Qwen is thinking..."):

            try:

                response = chain.invoke({
                    "history": history_text,
                    "question": user_input
                })

                st.markdown(response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:

                st.error(f"Error generating response: {e}")

