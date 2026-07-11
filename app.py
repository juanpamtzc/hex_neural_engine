import streamlit as st

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Hex-Zero Agentic Arena",
    page_icon="⬢",
    layout="wide"
)

st.title("⬢ Hex-Zero: Deep Reinforcement Learning Arena")
st.caption("Architected by a Staff MLE | Built with AlphaZero Search & PyTorch Inference")

# --- 2. FLAWLESS STATE MANAGEMENT ---
# Initialize the chat history for the LLM
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Hex-Zero Arena. I am your AI assistant. I can help you analyze the board, query the documentation, or execute engine moves. How can I assist you?"}
    ]

# Initialize the LLM / Model loaded state to prevent reloading
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Initialize the Hex Board (stubbed state for the engine tool later)
if "board_state" not in st.session_state:
    st.session_state.board_state = None  

# --- 3. UI SKELETON: SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Engine Status")
    
    # We will use this in Step 2 to trigger the @st.cache_resource model loading
    if not st.session_state.model_loaded:
        st.warning("LLM Pipeline Offline")
        if st.button("Initialize PyTorch Models", use_container_width=True):
            with st.spinner("Loading models into RTX 4070 VRAM..."):
                # TODO: model_pipeline.py invocation goes here
                st.session_state.model_loaded = True
                st.rerun()
    else:
        st.success("LLM Pipeline Active")
        
    st.divider()
    if st.button("Reset Session", type="primary", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Session reset. How can I help you?"}
        ]
        st.session_state.board_state = None
        st.rerun()

# --- 4. UI SKELETON: CHAT INTERFACE ---
# Render the existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture user input
if prompt := st.chat_input("Ask a question or command the engine..."):
    # 1. Append user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Render user message to UI immediately
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 3. Render Assistant response skeleton
    with st.chat_message("assistant"):
        # We will replace this with actual LLM generation and st.status tool-calls in Step 3
        with st.spinner("Thinking..."):
            # Mocking the pipeline response for now
            response = f"I received your command: '{prompt}'. *(LLM logic pending Step 2/3)*"
            st.markdown(response)
            
    # 4. Append assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": response})