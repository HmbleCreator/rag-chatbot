import os
import logging
import json
import time
from llama_cpp import Llama
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.model_config import ModelConfig

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
        
    def check_system_memory(self, required_mb=4000):
        """Check if system has enough available memory"""
        try:
            import psutil
            available = psutil.virtual_memory().available / (1024 * 1024)
            if available < required_mb:
                logger.warning(f"Low memory: Only {available:.0f}MB available, {required_mb}MB recommended")
                return False
            return True
        except ImportError:
            logger.warning("psutil not installed, skipping memory check")
            return True
        
    def load_model(self) -> None:
        """Safely load the GGUF model with fallback options"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Check available memory and adjust context if needed
        if not self.check_system_memory():
            self.model_config['n_ctx'] = min(self.model_config['n_ctx'], 1024)
            self.model_config['n_threads'] = min(self.model_config['n_threads'], 4)
        
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
                logger.exception("Exception details:")
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
            
            start_time = time.time()
            
            # Improved generation parameters
            response = self.model.create_completion(
                formatted_prompt,
                max_tokens=2048,  # Increased for longer responses
                temperature=0.3,  # Lower temperature for more focused responses
                top_p=0.92,
                repeat_penalty=1.2,  # Stronger repeat penalty
                top_k=50,
                echo=False,
                stop=["User:", "\n\nUser:", "Current Question:"]  # Better stop sequences
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Response generated in {generation_time:.2f} seconds")
            
            if not response or not response['choices']:
                raise RuntimeError("Model returned empty response")
                
            response_text = response['choices'][0]['text'].strip()
            
            # Verify and potentially improve the response
            if len(response_text) < 100 and "summary" in prompt.lower():
                response_text = self.expand_incomplete_response(prompt, context, response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise

    def _format_prompt(self, 
                      prompt: str, 
                      context: Optional[str] = None, 
                      chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Format the prompt with context and chat history"""
        parts = []
        
        # More detailed system message
        parts.append("System: You are a helpful assistant that provides accurate, complete, and well-structured responses based on the provided document context. Always answer questions thoroughly and completely, and when summarizing content, include all key points from the relevant documents.")
        
        # Add context with better framing
        if context:
            parts.append(f"Context (relevant document excerpts for your reference):\n{context}")
        
        # Add chat history
        if chat_history and len(chat_history) > 0:
            parts.append("Previous conversation:")
            for exchange in chat_history:
                parts.append(f"User: {exchange['user']}")
                parts.append(f"Assistant: {exchange['assistant']}")
        
        # Add current prompt with better framing
        parts.append(f"Current Question: {prompt}")
        parts.append("Assistant's Complete Response:")
        
        return "\n\n".join(parts)

    def expand_incomplete_response(self, prompt, context, initial_response):
        """Expand responses that appear incomplete"""
        logger.info("Detected potentially incomplete response. Attempting to expand it.")
        
        expand_prompt = f"""
        System: The previous response was incomplete. Please provide a complete and thorough response to the user's question.
        
        Context: {context}
        
        Question: {prompt}
        
        Incomplete response: {initial_response}
        
        Complete and thorough response:
        """
        
        try:
            expanded = self.model.create_completion(
                expand_prompt,
                max_tokens=2048,
                temperature=0.3,
                top_p=0.95,
                repeat_penalty=1.2,
                echo=False
            )
            
            if expanded and expanded['choices']:
                expanded_text = expanded['choices'][0]['text'].strip()
                if len(expanded_text) > len(initial_response):
                    logger.info("Successfully expanded the response")
                    return expanded_text
            
            logger.warning("Failed to expand the response, returning original")
            return initial_response
            
        except Exception as e:
            logger.error(f"Error expanding response: {str(e)}")
            return initial_response

    def update_config(self, **kwargs) -> None:
        """Update model configuration"""
        self.model_config.update(kwargs)
        self.model = None  # Force model reload with new config
        logger.info(f"Updated configuration: {json.dumps(self.model_config, indent=2)}")