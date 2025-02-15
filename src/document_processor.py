from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, pdf_dir, vector_store_path):
        self.pdf_dir = pdf_dir
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        
        # Create necessary directories
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
        
        logger.info(f"Initialized DocumentProcessor with PDF dir: {pdf_dir}")
        logger.info(f"Vector store path: {vector_store_path}")
    
    def list_processed_pdfs(self):
        """List all processed PDFs"""
        if os.path.exists(self.pdf_dir):
            pdfs = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
            logger.info(f"Found {len(pdfs)} processed PDFs: {pdfs}")
            return pdfs
        return []
    
    def update_vector_store(self, new_pdf_path):
        """Add new PDF to vector store"""
        try:
            logger.info(f"Processing new PDF: {new_pdf_path}")
            splits = self.process_pdf(new_pdf_path)
            logger.info(f"Created {len(splits)} splits from PDF")
            
            vector_store_file = os.path.join(self.vector_store_path, "index.faiss")
            if not os.path.exists(vector_store_file):
                logger.info("Creating new vector store")
                self.vector_store = FAISS.from_documents(
                    splits,
                    self.embeddings
                )
            else:
                logger.info("Loading existing vector store")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Adding new documents to vector store")
                self.vector_store.add_documents(splits)
            
            logger.info("Saving updated vector store")
            self.vector_store.save_local(self.vector_store_path)
            return True
            
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            raise

    def process_pdf(self, pdf_path):
        """Process PDF and create document splits"""
        try:
            logger.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Get filename for metadata
            filename = os.path.basename(pdf_path)
            
            # Better splitting strategy
            splitter = CharacterTextSplitter(
                chunk_size=1500,  # Larger chunks for better context
                chunk_overlap=200,  # More overlap to prevent missing context
                separator="\n"
            )
            splits = splitter.split_documents(docs)
            
            # Enhance metadata
            for split in splits:
                if 'source' in split.metadata:
                    split.metadata['source'] = pdf_path
                split.metadata['filename'] = filename
            
            logger.info(f"Created {len(splits)} chunks from PDF")
            return splits
        
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def get_relevant_context(self, query, k=3):
        """Retrieve relevant context for a query"""
        try:
            logger.info(f"Getting context for query: {query}")
            
            # Check if vector store exists
            vector_store_file = os.path.join(self.vector_store_path, "index.faiss")
            if not os.path.exists(vector_store_file):
                logger.warning("No vector store found!")
                return "No documents have been processed yet. Please upload a PDF first."
            
            # Load vector store if needed
            if self.vector_store is None:
                logger.info("Loading vector store")
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            
            # Increase number of docs retrieved for important queries
            if any(keyword in query.lower() for keyword in ['name', 'who', 'person', 'resume']):
                k = 5
            
            # Get similar documents
            docs = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(docs)} relevant documents")
            
            # Better context formatting with document metadata
            context_parts = []
            for i, doc in enumerate(docs):
                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                page = doc.metadata.get('page', 'unknown page')
                context_parts.append(f"[Document {i+1}: {source}, Page {page}]\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            # Add a prefix to help the model understand the context
            context = f"RELEVANT DOCUMENT EXCERPTS:\n{context}"
            
            logger.info(f"Returning context of length: {len(context)}")
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            raise