import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, SystemMessage
import uuid

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #8B4513;
        text-align: center;
        margin-bottom: 1rem;
    }
    .mudra-info {
        background-color: #FFF8DC;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #DAA520;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E6F3FF;
        border-left: 5px solid #0074D9;
    }
    .assistant-message {
        background-color: #F0F8FF;
        border-left: 5px solid #2ECC40;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if 'expert_mode' not in st.session_state:
    st.session_state.expert_mode = False

# Header
st.markdown('<div class="main-header">🪷 Mudra AI Assistant</div>', unsafe_allow_html=True)

# Sidebar with mudra information and controls
with st.sidebar:
    st.header("🎭 Mudra Assistant")
    
    st.markdown("""
    **Common Mudras to ask about:**
    - Pataka (flag)
    - Tripataka (three-part flag)
    - Ardhapataka (half flag)
    - Kartarimukha (scissors)
    - Mayura (peacock)
    - Ardhachandra (half moon)
    - Shikhara (spire)
    - Kapitta (elephant apple)
    - Kataka (link)
    - Suchi (needle)
    """)
    
    st.divider()
    
    # Expert mode toggle
    expert_mode = st.toggle("Expert Mode", value=st.session_state.expert_mode)
    st.session_state.expert_mode = expert_mode
    
    if st.button("🆕 New Conversation"):
        st.session_state.message_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    if st.button("📚 Sample Questions"):
        sample_questions = [
            "Explain the Pataka mudra",
            "What are the uses of Tripataka in Bharatanatyam?",
            "Show me mudras for expressing emotions",
            "Difference between Ardhapataka and Kartarimukha"
        ]
        for question in sample_questions:
            if st.button(question, use_container_width=True):
                st.session_state.user_input = question

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Mudra Conversation")
    
    # Display chat history with better formatting
    for message in st.session_state.message_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>Mudra Expert:</strong><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.subheader("Quick Actions")
    if st.button("🔄 Refresh"):
        st.rerun()
    
    if st.button("💾 Save Chat"):
        st.success("Conversation saved!")

# Chat input
user_input = st.chat_input("Ask about dance mudras...")

if user_input:
    # Add user message to history
    st.session_state.message_history.append({"role": "user", "content": user_input})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
    
    # Prepare messages for the chatbot
    messages = [HumanMessage(content=user_input)]
    
    # Add expert mode context if enabled
    if st.session_state.expert_mode:
        expert_prompt = "Provide detailed technical analysis including muscle movements, historical context, and advanced variations."
        messages.insert(0, SystemMessage(content=expert_prompt))
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing mudras..."):
            try:
                response = chatbot.invoke(
                    {"messages": messages}, 
                    config={"configurable": {"thread_id": st.session_state.thread_id}}
                )
                ai_response = response['messages'][-1].content
                
                # Display response
                st.write(ai_response)
                
                # Add to history
                st.session_state.message_history.append({
                    "role": "assistant", 
                    "content": ai_response
                })
                
            except Exception as e:
                error_msg = "I apologize, but I encountered an error. Please try again."
                st.error(error_msg)
                st.session_state.message_history.append({
                    "role": "assistant", 
                    "content": error_msg
                })

# Footer with mudra information
st.markdown("---")
st.markdown("""
<div class="mudra-info">
    <strong>Did you know?</strong> There are 28 single-hand mudras (Asamyukta Hastas) and 
    24 combined-hand mudras (Samyukta Hastas) in Bharatanatyam. Each mudra can represent 
    various objects, gods, relationships, and actions when used in different contexts.
</div>
""", unsafe_allow_html=True)