import os
from typing import Optional

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from config import GROQ_API_KEY

class RAGApplication:
    def __init__(
        self, 
        pdf_path: str, 
        groq_api_key: Optional[str] = None,
        embed_model: Optional[BaseEmbedding] = None,
        llm_model: str = 'mixtral-8x7b-32768',
        storage_dir: Optional[str] = None
    ):
        """
        Initialize RAG application with PDF embedding and Groq LLM
        
        :param pdf_path: Path to the PDF document to embed
        :param groq_api_key: Groq API key (optional)
        :param embed_model: Optional embedding model (defaults to HuggingFace)
        :param llm_model: Groq LLM model to use
        :param storage_dir: Directory to store vector index (defaults to current working directory)
        """
        # Retrieve Groq API Key from multiple sources
        self.groq_api_key = self._get_groq_api_key(groq_api_key)
        
        # Set up embedding model
        self.embed_model = embed_model or HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Set up LLM
        self.llm = Groq(
            api_key=self.groq_api_key, 
            model=llm_model
        )
        
        # PDF and storage paths
        self.pdf_path = pdf_path
        
        # Use current working directory if no storage dir specified
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), 'rag_index')
        
        # Ensure storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Load the existing vector store index if available
        self.index = self.load_index()

    def _get_groq_api_key(self, provided_key: Optional[str] = None) -> str:
        """
        Retrieve Groq API Key from multiple sources
        
        Priority order:
        1. Directly provided key
        2. Environment variable
        3. Raise error if no key found
        
        :param provided_key: API key passed directly to the constructor
        :return: Groq API Key
        """
        # Check if key was directly provided
        if provided_key:
            return provided_key
        
        # Check environment variable
        env_key = os.getenv('GROQ_API_KEY')
        if env_key:
            return env_key
        
        # Check configuration file (optional enhancement)
        try:
            with open('.groq_api_key', 'r') as f:
                file_key = f.read().strip()
                if file_key:
                    return file_key
        except FileNotFoundError:
            pass
        
        # Prompt user for API key if no other method works
        api_key = input("Please enter your Groq API Key: ").strip()
        if not api_key:
            raise ValueError("No Groq API Key provided. Cannot initialize RAG application.")
        
        # Optional: Save to .env file for future use
        self._save_api_key(api_key)
        
        return api_key
    
    def _save_api_key(self, api_key: str):
        """
        Optionally save API key to a configuration file
        
        :param api_key: Groq API Key to save
        """
        try:
            with open('.groq_api_key', 'w') as f:
                f.write(api_key)
            print("API key saved for future use. Keep this file secure!")
        except Exception as e:
            print(f"Could not save API key: {e}")
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Load existing vector store index if available
        
        :return: VectorStoreIndex or None if loading fails
        """
        try:
            # Rebuild storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=self.storage_dir
            )
            
            # Load index
            return load_index_from_storage(
                storage_context, 
                embed_model=self.embed_model,
                llm=self.llm
            )
        except FileNotFoundError:
            print("No existing index found. You need to create the index first.")
            return None
    
    def create_index(self) -> VectorStoreIndex:
        """
        Create a vector store index from the PDF document
        
        :return: VectorStoreIndex
        """
        # Read PDF document
        documents = SimpleDirectoryReader(input_files=[self.pdf_path]).load_data()
        
        # Create vector store index
        self.index = VectorStoreIndex.from_documents(
            documents, 
            embed_model=self.embed_model,
            llm=self.llm
        )
        
        # Persist the index
        self.index.storage_context.persist(persist_dir=self.storage_dir)
        
        return self.index
    
    def query(self, query_str: str, similarity_top_k: int = 5) -> str:
        """
        Query the RAG system
        
        :param query_str: Query string
        :param similarity_top_k: Number of top similar documents to retrieve
        :return: Generated response
        """
        if not self.index:
            raise ValueError("Index is not available. Please create the index first.")
        
        # Create query engine
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=similarity_top_k
        )
        
        # Execute query
        response = query_engine.query(query_str)
        
        return str(response)

def main():
    # Example usage with flexible API key handling
    groq_api_key = GROQ_API_KEY
    pdf_path = 'data/The Big Book of Dashboards.pdf'
    
    # Option 1: No custom storage directory (uses current working directory)
    rag_app = RAGApplication(pdf_path=pdf_path, groq_api_key=groq_api_key)
    
    # Option 2: Specify a custom storage directory
    # rag_app = RAGApplication(
    #     pdf_path=pdf_path, 
    #     groq_api_key=groq_api_key, 
    #     storage_dir='/path/to/custom/directory'
    # )
    
    # Example query
    query = "Who wrote the Big Book of Dashboards?"
    response = rag_app.query(query)
    print(response)

if __name__ == "__main__":
    main()
