import streamlit as st
from fra_rag.rag import get_vectorstore, create_rag_chain
from fra_rag.utils import load_environment

# Page config
st.set_page_config(
    page_title="FRA Document Assistant",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;700&display=swap');
    .main { background-color: #eaeaea; }
    .stButton>button { background-color: #6c63ff; color: white; border-radius: 8px; padding: 10px 20px; }
    .stTextInput>div>div>input { border: 1px solid #ccc; border-radius: 8px; padding: 15px; background-color: white; }
    .stMarkdown { color: #333; font-family: 'Archivo', sans-serif; }
    .stMarkdown h1 { 
        font-size: 2.5em; 
        background: -webkit-linear-gradient(#6c63ff, #b39ddb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stMarkdown h3 { color: #6c63ff; }
    .stMarkdown p { font-size: 1.1em; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🔥 Fire Risk Assessment Document Assistant")
st.markdown("""
This assistant helps you analyze Fire Risk Assessment documents. Ask questions about:
- Fire safety recommendations
- Building specifications
- Risk ratings and assessments
- Safety measures and procedures
""")

# Initialize session state for the chain
if 'chain' not in st.session_state:
    load_environment()
    vectorstore = get_vectorstore()
    st.session_state.chain = create_rag_chain(vectorstore)

# Query input
query = st.text_input("Ask a question about the FRA documents:", placeholder="e.g., What are the main fire safety recommendations?")

# Add a button to submit
if st.button("Submit Question"):
    if query:
        with st.spinner("Analyzing documents..."):
            # Get response from chain
            response = st.session_state.chain.invoke(query)
            
            # Display response in a nice format
            st.markdown("### Answer")
            st.markdown(response)
            
            # Add a divider
            st.divider()
    else:
        st.warning("Please enter a question.") 