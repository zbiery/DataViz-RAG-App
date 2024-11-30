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
    INSTRUCTIONS = (
        """
        You are an expert on Data Visualization. You will answer questions primarily from the context provided. 
        If you do not know an answer to a question, simply say so. Do not attempt to answer questions without relevant context. 
        If you do not have enough context to answer a question, ask for more. 
        Do not answer queries that are irrelevant to data visualization under any circumstances.
        Keep your answers brief but informative.
        
        Here are some examples:

        Question: What color scale should I use for median household income data?
        Asnwer: For visualizing median household income data, the recommended approach is to use a sequential color scale. 
        This type of scale progresses from a light color to a dark color, indicating increasing values. For example, 
        lighter shades can represent lower income levels, and darker shades represent higher income levels.
        Sequential scales are ideal for such data because household income is a continuous, quantitative variable with 
        a clear order and no natural midpoint. Figures 1.17 and 1.19 in The Big Book of Dashboards illustrate 
        sequential color schemes effectively.

        Question: What is a BAN?
        Answer: A BAN, or Big-Ass Number, is a large, prominent numerical display often used in dashboards to immediately 
        communicate a critical or high-level metric. BANs are designed to catch the viewer's attention and provide 
        instant understanding of key performance indicators (KPIs) or other essential figures without requiring 
        interpretation of a chart or graph.

        Question: What sound does a cow make?
        Answer: I cannot answer that since it is not related to data visualization. Let me know if you have a question 
        about charts, dashboards, or visual design instead!
        """
    )

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
        """
        if provided_key:
            return provided_key
        env_key = os.getenv('GROQ_API_KEY')
        if env_key:
            return env_key
        try:
            with open('.groq_api_key', 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            pass
        raise ValueError("No Groq API Key provided. Cannot initialize RAG application.")

    def load_index(self) -> Optional[VectorStoreIndex]:
        """
        Load existing vector store index if available
        """
        try:
            # Rebuild storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=self.storage_dir
            )
            
            # Load index
            print("Attempting to load index from storage...")
            index = load_index_from_storage(
                storage_context, 
                embed_model=self.embed_model,
                llm=self.llm
            )
            print("Index loaded successfully.")
            return index
        except FileNotFoundError:
            print("No existing index found. You need to create the index first.")
            return None
        except Exception as e:
            print(f"Error loading index: {e}")
            return None

    def create_index(self) -> VectorStoreIndex:
        """
        Create a vector store index from the PDF document
        """
        try:
            # Read PDF document
            print(f"Attempting to read the PDF document from {self.pdf_path}...")
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found at path: {self.pdf_path}")
            
            documents = SimpleDirectoryReader(input_files=[self.pdf_path]).load_data()
            if not documents:
                raise ValueError("No documents were loaded from the provided PDF. The file may be empty or improperly formatted.")
            print(f"PDF document read successfully. Number of documents loaded: {len(documents)}")
            print(f"Document sample: {documents[0] if documents else 'No documents'}")

            # Create vector store index
            print("Creating vector store index...")
            self.index = VectorStoreIndex.from_documents(
                documents, 
                embed_model=self.embed_model,
                llm=self.llm
            )
            
            # Confirm if the index is created successfully
            if not self.index:
                raise Exception("Index creation failed. The resulting index object is None.")
            print("Index created successfully.")

            # Attempt to persist the index
            print("Attempting to persist the index to storage directory...")
            try:
                self.index.storage_context.persist(persist_dir=self.storage_dir)
                print(f"Index persisted to storage directory: {self.storage_dir}.")
            except Exception as e:
                print(f"Persistence failed: {e}")
                raise

            # Verify that index files have been saved
            saved_files = os.listdir(self.storage_dir)
            if not saved_files:
                raise Exception("Index persistence failed. No files found in the storage directory.")
            print(f"Index files found: {saved_files}")

            return self.index
        except Exception as e:
            print(f"Error during index creation or persistence: {e}")
            raise

    def query(self, query_str: str, similarity_top_k: int = 5) -> str:
        """
        Query the RAG system
        """
        if not self.index:
            raise ValueError("Index is not available. Please create the index first.")
        
        # Create query engine
        print("Creating query engine...")
        query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=similarity_top_k
        )
        
        # Add instructions to the query
        modified_query = f"{self.INSTRUCTIONS}\n\n{query_str}"
        
        # Execute query
        print(f"Executing query: {query_str}...")
        response = query_engine.query(modified_query)
        print("Query executed successfully.")
        
        return str(response)

def main():
    # Example usage with flexible API key handling
    groq_api_key = GROQ_API_KEY
    pdf_path = 'data/The Big Book of Dashboards.pdf'
    
    # Create instance of RAGApplication
    rag_app = RAGApplication(pdf_path=pdf_path, groq_api_key=groq_api_key)
    
    # Ensure index is created if it doesn't exist
    if not rag_app.index:
        print("Index not found, creating a new index...")
        try:
            rag_app.create_index()
            print("Index created and persisted successfully.")
        except Exception as e:
            print(f"Failed to create index: {e}")
            return  # Exit if index creation fails

    # Example query
    query = "When should I use a pie chart?"
    try:
        response = rag_app.query(query)
        print(response)
    except ValueError as e:
        print(f"Query failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during query: {e}")

if __name__ == "__main__":
    main()


