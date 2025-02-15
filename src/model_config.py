import os
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class ModelConfig:
    DEFAULT_CONFIG = {
        'mistral-7b': {
            'n_ctx': 4096,
            'n_threads': os.cpu_count(),
            'n_gpu_layers': 0,
            'template': "System: You are a helpful assistant.\n\nUser: {query}\n\nAssistant:"
        },
        'llama-2': {
            'n_ctx': 4096,
            'n_threads': os.cpu_count(),
            'n_gpu_layers': 0,
            'template': "<s>[INST] {query} [/INST]"
        },
        'neural-chat': {
            'n_ctx': 4096,
            'n_threads': os.cpu_count(),
            'n_gpu_layers': 0,
            'template': "### System: You are a helpful assistant.\n\n### User: {query}\n\n### Assistant:"
        },
        'mistral-nemo-instruct-2407-gguf': {
            'n_ctx': 4096,
            'n_threads': os.cpu_count(),
            'n_gpu_layers': 0,
            'template': "System: You are a helpful assistant.\n\nUser: {query}\n\nAssistant:"
        },
        'model': {
            'n_ctx': 4096,
            'n_threads': os.cpu_count(),
            'n_gpu_layers': 0,
            'template': "System: You are a helpful assistant.\n\nUser: {query}\n\nAssistant:"
        }
    }

    @classmethod
    def get_config(cls, model_name: str) -> Dict:
        """Get configuration for a specific model"""
        base_config = cls.DEFAULT_CONFIG.get(model_name, cls.DEFAULT_CONFIG['mistral-7b'])
        return base_config.copy()

    @classmethod
    def detect_model_type(cls, model_path: str) -> Optional[str]:
        """Detect model type from filename"""
        filename = os.path.basename(model_path).lower()

        if 'mistral-nemo-instruct-2407-gguf' in filename:
            return 'mistral-nemo-instruct-2407-gguf'
        elif 'mistral' in filename:
            return 'mistral-7b'
        elif 'llama-2' in filename:
            return 'llama-2'
        elif 'neural-chat' in filename:
            return 'neural-chat'
        elif 'model.gguf' in filename:
            return 'model'
        else:
            return None

    @classmethod
    def save_custom_config(cls, model_name: str, config: Dict) -> None:
        """Save custom configuration for a model"""
        config_path = os.path.join(os.path.dirname(__file__), 'model_configs.json')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    configs = json.load(f)
            else:
                configs = {}

            configs[model_name] = config

            with open(config_path, 'w') as f:
                json.dump(configs, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving custom config: {str(e)}")

from llama_cpp import Llama
from typing import List

class ModelHandler:
    def __init__(self, model_path: str):
        """
        Initialize the model handler
        model_path: Path to the GGUF model file
        """
        self.model_path = model_path
        self.model = None
        self.model_config = self.load_model_config()

    def load_model_config(self) -> Dict:
        """Load model configuration based on the model path"""
        model_name = ModelConfig.detect_model_type(self.model_path)
        if model_name is None:
            raise ValueError(f"Unable to detect model type from path: {self.model_path}")
        return ModelConfig.get_config(model_name)

    def load_model(self) -> None:
        """Safely load the GGUF model with fallback options"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        fallback_configs = [
            self.model_config,
            {**self.model_config, 'n_ctx': 2048, 'n_threads': min(10, self.model_config['n_threads'])},
            {**self.model_config, 'n_ctx': 1024, 'n_threads': 4}
        ]

        for config in fallback_configs:
            try:
                logger.info(f"Attempting to load model with config: {json.dumps(config, indent=2)}")
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=config['n_ctx'],
                    n_threads=config['n_threads'],
                    n_gpu_layers=config['n_gpu_layers'],
                    verbose=False  # Reduce overhead
                )
                logger.info("Model loaded successfully")
                self.model_config = config  # Update with working config
                return

            except Exception as e:
                logger.warning(f"Failed to load model with config {config}: {str(e)}")
                continue

        # If we get here, all attempts failed
        logger.error("All attempts to load model failed")
        raise RuntimeError("Unable to load model with any configuration")

    def generate_response(self, 
                          prompt: str, 
                          context: Optional[str] = None, 
                          chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate a response using the loaded model
        """
        try:
            if self.model is None:
                self.load_model()

            # Construct the prompt with context and chat history
            formatted_prompt = self._format_prompt(prompt, context, chat_history)
            
            # Generate response
            response = self.model.create_completion(
                formatted_prompt,
                max_tokens=2048,  # Increase max tokens for longer responses
                temperature=0.7,
                top_p=0.95,
                repeat_penalty=1.1,
                top_k=40,
                echo=False,
                stop=["User:", "Assistant:", "\n\n"]  # Adjust stop sequences
            )
            
            if not response or not response['choices']:
                raise RuntimeError("Model returned empty response")
                
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise

    def _format_prompt(self, 
                       prompt: str, 
                       context: Optional[str] = None, 
                       chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the prompt with context and chat history"""
        parts = []
        
        # Add system message
        parts.append("System: You are a helpful assistant that provides accurate and relevant information based on the provided context. Your responses should be clear and directly address the user's questions.")
        
        # Add context if provided
        if context:
            parts.append(f"Context: {context}")
        
        # Add chat history
        if chat_history:
            for exchange in chat_history:
                parts.append(f"User: {exchange['user']}")
                parts.append(f"Assistant: {exchange['assistant']}")
        
        # Add current prompt
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")  # Explicitly guiding the model to start the assistant's response
        
        return "\n\n".join(parts)

    def update_config(self, **kwargs) -> None:
        """Update model configuration"""
        self.model_config.update(kwargs)
        self.model = None  # Force model reload with new config
        logger.info(f"Updated configuration: {json.dumps(self.model_config, indent=2)}")
