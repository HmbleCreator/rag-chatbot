from flask import Flask, request, jsonify, render_template, flash
from werkzeug.utils import secure_filename
import os
import uuid
import time
import json
import logging
from datetime import datetime
from src.model_handler import ModelHandler
from src.model_config import ModelConfig
from src.document_processor import DocumentProcessor

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = Flask(__name__)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PDF_DIR = os.path.join(BASE_DIR, 'data', 'pdfs')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'data', 'vector_store')
CHAT_HISTORY_DIR = os.path.join(BASE_DIR, 'data', 'chat_history')

# Create necessary directories
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize document processor
doc_processor = DocumentProcessor(PDF_DIR, VECTOR_STORE_PATH)

# Initialize chat manager
class ChatManager:
    def __init__(self, history_dir):
        self.history_dir = history_dir
        logger.info(f"Initializing chat manager with history dir: {history_dir}")
        
    def create_chat(self):
        """Create a new chat session"""
        chat_id = str(uuid.uuid4())
        self.save_chat_history(chat_id, [])
        logger.info(f"Created new chat with ID: {chat_id}")
        return chat_id
        
    def get_chat_history(self, chat_id):
        """Get chat history for a specific chat"""
        try:
            history_file = os.path.join(self.history_dir, f"{chat_id}.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading chat history: {str(e)}")
            return []
            
    def save_chat_history(self, chat_id, history):
        """Save chat history for a specific chat"""
        try:
            history_file = os.path.join(self.history_dir, f"{chat_id}.json")
            with open(history_file, 'w') as f:
                json.dump(history, f)
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")

chat_manager = ChatManager(CHAT_HISTORY_DIR)

@app.route('/')
def home():
    # Create new chat session if none exists
    chat_id = chat_manager.create_chat()
    
    # List all available models
    available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.gguf')]
    return render_template('index.html', chat_id=chat_id, available_models=available_models)

@app.route('/new-chat', methods=['POST'])
def new_chat():
    """Start a new chat session"""
    chat_id = chat_manager.create_chat()
    return jsonify({'chat_id': chat_id})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        chat_id = data.get('chat_id')
        if not chat_id:
            return jsonify({'error': 'No chat ID provided'}), 400
        
        # Get chat history
        chat_history = chat_manager.get_chat_history(chat_id)
        
        # Get relevant context from vector store
        context = doc_processor.get_relevant_context(query)
        
        # Log the context being used
        logger.info(f"Context length: {len(context)}")
        logger.debug(f"Using context: {context}")
        
        # Generate response using the selected model
        model_path = data.get('model_path')
        if not model_path:
            return jsonify({'error': 'No model selected'}), 400
        model_handler = ModelHandler(model_path)
        
        start_time = time.time()
        response = model_handler.generate_response(query, context, chat_history)
        generation_time = time.time() - start_time
        
        logger.info(f"Total response time: {generation_time:.2f} seconds")
        
        # Update chat history
        chat_history.append({
            'user': query,
            'assistant': response
        })
        chat_manager.save_chat_history(chat_id, chat_history)
        
        # Log response for debugging
        logger.info(f"Response length: {len(response)}")
        logger.debug(f"Generated response: {response[:200]}...")
        
        return jsonify({
            'response': response,
            'chat_id': chat_id,
            'generation_time': round(generation_time, 2)
        })
    except json.JSONDecodeError:
        logger.error("Invalid JSON in request")
        return jsonify({'error': 'Invalid JSON data'}), 400
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/select-model', methods=['POST'])
def select_model():
    """Select model for the current session"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        model_filename = data.get('model_filename')
        if not model_filename:
            return jsonify({'error': 'No model filename provided'}), 400
            
        model_path = os.path.join(MODEL_DIR, model_filename)
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model not found'}), 400
            
        return jsonify({'model_path': model_path})
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    """Get chat history for current session"""
    chat_id = request.args.get('chat_id')
    if not chat_id:
        return jsonify({'error': 'No active chat session'}), 400
        
    history = chat_manager.get_chat_history(chat_id)
    return jsonify({'history': history})

@app.route('/debug/system-state', methods=['GET'])
def system_state():
    """Get current system state for debugging"""
    try:
        pdfs = doc_processor.list_processed_pdfs()
        vector_store_exists = os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss"))
        model_exists = any(f.endswith('.gguf') for f in os.listdir(MODEL_DIR))
        
        state = {
            'processed_pdfs': pdfs,
            'vector_store_exists': vector_store_exists,
            'model_exists': model_exists,
            'vector_store_path': VECTOR_STORE_PATH,
            'pdf_dir': PDF_DIR,
            'model_dir': MODEL_DIR
        }
        
        return jsonify(state)
    except Exception as e:
        logger.error(f"Error getting system state: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/debug/test-retrieval', methods=['POST'])
def test_retrieval():
    """Test document retrieval"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        query = data.get('query', '')
        context = doc_processor.get_relevant_context(query)
        return jsonify({
            'query': query,
            'context': context,
            'context_length': len(context)
        })
    except json.JSONDecodeError:
        logger.error("Invalid JSON in test-retrieval request")
        return jsonify({'error': 'Invalid JSON data'}), 400
    except Exception as e:
        logger.error(f"Error testing retrieval: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """Upload a new PDF"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        files = request.files.getlist('pdf')
        if not files:
            return jsonify({'error': 'No selected files'}), 400

        uploaded_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(PDF_DIR, filename)
                file.save(file_path)
                uploaded_files.append(file_path)

                # Process PDF and update vector store
                doc_processor.update_vector_store(file_path)

        return jsonify({'message': 'Files successfully uploaded and processed', 'files': uploaded_files}), 200
        
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Check if the uploaded file is a PDF"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

@app.route('/debug/document-info', methods=['GET'])
def document_info():
    """Get detailed information about processed documents"""
    try:
        pdfs = doc_processor.list_processed_pdfs()
        pdf_info = []

        for pdf in pdfs:
            pdf_path = os.path.join(PDF_DIR, pdf)
            file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB

            info = {
                'filename': pdf,
                'path': pdf_path,
                'size_mb': round(file_size, 2),
                'last_modified': str(datetime.fromtimestamp(os.path.getmtime(pdf_path)))
            }
            pdf_info.append(info)

        return jsonify({
            'total_documents': len(pdfs),
            'documents': pdf_info,
            'vector_store_path': VECTOR_STORE_PATH,
            'vector_store_exists': os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss"))
        })
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
