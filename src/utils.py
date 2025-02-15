# src/utils.py
import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    context_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_threads: int = 4
    repeat_penalty: float = 1.1
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ModelConfig':
        """Load configuration from YAML file"""
        if not os.path.exists(path):
            logger.warning(f"Config file {path} not found, using defaults")
            return cls()
            
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep punctuation
        text = ''.join(char for char in text if char.isprintable())
        return text.strip()
    
    @staticmethod
    def format_prompt(query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt with context and optional system prompt"""
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
            
        if context:
            parts.append(f"Context: {context}")
            
        parts.append(f"User: {query}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)

class DocumentManager:
    """Manage document processing and tracking"""
    
    def __init__(self, doc_dir: str):
        self.doc_dir = doc_dir
        self.doc_index_path = os.path.join(doc_dir, 'doc_index.json')
        self.doc_index = self._load_doc_index()
        
    def _load_doc_index(self) -> Dict:
        """Load document index from JSON"""
        if os.path.exists(self.doc_index_path):
            with open(self.doc_index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_doc_index(self):
        """Save document index to JSON"""
        with open(self.doc_index_path, 'w') as f:
            json.dump(self.doc_index, f, indent=2)
    
    def add_document(self, filename: str, metadata: Dict = None) -> None:
        """Add document to index with metadata"""
        doc_id = str(len(self.doc_index) + 1)
        self.doc_index[doc_id] = {
            'filename': filename,
            'added_at': str(datetime.now()),
            'metadata': metadata or {}
        }
        self._save_doc_index()
        
    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """Get document information from index"""
        return self.doc_index.get(doc_id)
    
    def list_documents(self) -> List[Dict]:
        """List all documents with their metadata"""
        return [
            {'id': k, **v}
            for k, v in self.doc_index.items()
        ]

class ModelManager:
    """Manage model configuration and performance monitoring"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config = ModelConfig.from_yaml(config_path) if config_path else ModelConfig()
        self.metrics = {
            'total_queries': 0,
            'avg_response_time': 0,
            'error_count': 0
        }
    
    def update_metrics(self, response_time: float, error: bool = False):
        """Update performance metrics"""
        self.metrics['total_queries'] += 1
        if error:
            self.metrics['error_count'] += 1
        
        # Update running average of response time
        prev_avg = self.metrics['avg_response_time']
        n = self.metrics['total_queries']
        self.metrics['avg_response_time'] = (
            (prev_avg * (n - 1) + response_time) / n
        )
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            **self.metrics,
            'error_rate': (
                self.metrics['error_count'] / self.metrics['total_queries']
                if self.metrics['total_queries'] > 0 else 0
            )
        }
    
    def get_model_config(self) -> Dict:
        """Get current model configuration"""
        return {
            'model_path': self.model_path,
            **self.config.__dict__
        }