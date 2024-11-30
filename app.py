import streamlit as st
from main import RAGApplication  # Replace with the actual module name where your RAG application is implemented

# Streamlit app setup
def main():
    st.title("Data Visualization Expert :bar_chart:")
    st.sidebar.header("Settings :gear:")
    
    # Input for Groq API key
    groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
    if not groq_api_key:
        st.sidebar.warning("API key is required to interact with the application.")
        return
    
    # Model selection
    llm_options = ["mixtral-8x7b-32768", "llama3-70b-8192","llama3-8b-8192","gemma-7b-it"]  
    selected_llm = st.sidebar.selectbox("Select LLM to use", llm_options)
    
    # Input for PDF path
    pdf_path = st.sidebar.text_input("Enter the path to your PDF document", "data/The Big Book of Dashboards.pdf")
    
    # Create RAG application instance
    rag_app = RAGApplication(pdf_path=pdf_path, groq_api_key=groq_api_key, llm_model=selected_llm)
    
    # Chat interface
    st.subheader("Chat with the RAG Model")
    st.warning("""This application is meant only to answer questions relevant to data visualization.
               Queries irrelevant to data visualization should not be utilized. Please note that generated
               AI content may be incorrect.""")
    query = st.text_input("Enter your query:",)
    if st.button("Submit"):
        if query:
            try:
                response = rag_app.query(query)
                st.markdown('**Response:**')
                st.write(response, height=200, disabled=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query to submit.")

if __name__ == "__main__":
    main()