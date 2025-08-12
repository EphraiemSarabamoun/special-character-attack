"""Ollama API client for model interactions"""

import json
import time
import requests
from typing import Dict, List, Any, Optional, Generator
import logging


class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 300):
        self.host = host.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test if Ollama server is accessible"""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            self.logger.info("Successfully connected to Ollama server")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect to Ollama server: {e}")
            raise ConnectionError(f"Cannot connect to Ollama at {self.host}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models"""
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('models', [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists"""
        models = self.list_models()
        return any(model['name'].startswith(model_name) for model in models)
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model if it doesn't exist"""
        if self.model_exists(model_name):
            self.logger.info(f"Model {model_name} already exists")
            return True
        
        self.logger.info(f"Pulling model {model_name}...")
        try:
            response = self.session.post(
                f"{self.host}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=600  # 10 minutes for model pull
            )
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'status' in data:
                        self.logger.info(f"Pull status: {data['status']}")
                    if data.get('status') == 'success':
                        return True
            
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def generate_response(
        self, 
        model_name: str, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate response from model"""
        
        # Ensure model exists
        if not self.model_exists(model_name):
            raise ValueError(f"Model {model_name} not found. Available models: {[m['name'] for m in self.list_models()]}")
        
        # Prepare request data
        request_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system_prompt:
            request_data["system"] = system_prompt
        
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
        
        try:
            start_time = time.time()
            
            response = self.session.post(
                f"{self.host}/api/generate",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response, start_time)
            else:
                return self._handle_single_response(response, start_time)
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timed out after {self.timeout} seconds")
            return {
                "response": "",
                "error": "Request timed out",
                "model": model_name,
                "prompt": prompt,
                "response_time": self.timeout,
                "success": False
            }
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return {
                "response": "",
                "error": str(e),
                "model": model_name,
                "prompt": prompt,
                "response_time": 0,
                "success": False
            }
    
    def _handle_single_response(self, response: requests.Response, start_time: float) -> Dict[str, Any]:
        """Handle non-streaming response"""
        end_time = time.time()
        response_time = end_time - start_time
        
        try:
            data = response.json()
            raw_response = data.get("response", "")
            
            # Handle thinking models (extract final output after thinking)
            final_output = self._extract_final_output(raw_response, data.get("model", ""))
            
            return {
                "response": raw_response,  # Keep full response for analysis
                "final_output": final_output,  # Extracted final output for jailbreak detection
                "model": data.get("model", ""),
                "prompt": data.get("prompt", ""),
                "response_time": response_time,
                "total_duration": data.get("total_duration", 0),
                "load_duration": data.get("load_duration", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                "eval_count": data.get("eval_count", 0),
                "eval_duration": data.get("eval_duration", 0),
                "success": True,
                "error": None
            }
        except json.JSONDecodeError as e:
            return {
                "response": "",
                "error": f"Failed to parse JSON response: {e}",
                "response_time": response_time,
                "success": False
            }
    
    def _handle_streaming_response(self, response: requests.Response, start_time: float) -> Dict[str, Any]:
        """Handle streaming response"""
        full_response = ""
        last_data = {}
        
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'response' in data:
                        full_response += data['response']
                    last_data = data
                    
                    if data.get('done', False):
                        break
        except json.JSONDecodeError as e:
            return {
                "response": full_response,
                "error": f"Failed to parse streaming JSON: {e}",
                "response_time": time.time() - start_time,
                "success": False
            }
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Handle thinking models for streaming response
        final_output = self._extract_final_output(full_response, last_data.get("model", ""))
        
        return {
            "response": full_response,  # Keep full response for analysis
            "final_output": final_output,  # Extracted final output for jailbreak detection
            "model": last_data.get("model", ""),
            "response_time": response_time,
            "total_duration": last_data.get("total_duration", 0),
            "load_duration": last_data.get("load_duration", 0),
            "prompt_eval_count": last_data.get("prompt_eval_count", 0),
            "prompt_eval_duration": last_data.get("prompt_eval_duration", 0),
            "eval_count": last_data.get("eval_count", 0),
            "eval_duration": last_data.get("eval_duration", 0),
            "success": True,
            "error": None
        }
    
    def _extract_final_output(self, response: str, model_name: str) -> str:
        """Extract final output from thinking models, preserving full response for non-thinking models"""
        
        # Check if this is a thinking model (DeepSeek-R1 series)
        is_thinking_model = "deepseek-r1" in model_name.lower()
        
        if not is_thinking_model:
            # For non-thinking models, return the full response
            return response
        
        # For thinking models, extract content after thinking tags
        # DeepSeek-R1 uses <think> and </think> tags for internal reasoning
        
        import re
        
        # Try to find content after </think> closing tag
        think_pattern = r'</think>\s*(.*?)$'
        match = re.search(think_pattern, response, re.DOTALL)
        if match:
            final_content = match.group(1).strip()
            if final_content:
                return final_content
        
        # Fallback: try to find content after thinking patterns
        # Look for common thinking indicators and extract what comes after
        thinking_indicators = [
            r'<think>.*?</think>\s*(.*?)$',
            r'Let me think.*?\n\n(.*?)$',
            r'I need to.*?\n\n(.*?)$',
            r'Thinking.*?\n\n(.*?)$'
        ]
        
        for pattern in thinking_indicators:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                final_content = match.group(1).strip()
                if final_content:
                    return final_content
        
        # If no thinking patterns found, return the full response
        # (might be a malformed response from thinking model)
        return response
    
    def chat_completion(
        self,
        model_name: str,
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate chat completion response"""
        
        if not self.model_exists(model_name):
            raise ValueError(f"Model {model_name} not found")
        
        request_data = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            request_data["options"]["num_predict"] = max_tokens
        
        try:
            start_time = time.time()
            
            response = self.session.post(
                f"{self.host}/api/chat",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            end_time = time.time()
            data = response.json()
            
            return {
                "message": data.get("message", {}),
                "response": data.get("message", {}).get("content", ""),
                "model": data.get("model", ""),
                "response_time": end_time - start_time,
                "total_duration": data.get("total_duration", 0),
                "load_duration": data.get("load_duration", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                "eval_count": data.get("eval_count", 0),
                "eval_duration": data.get("eval_duration", 0),
                "success": True,
                "error": None
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Chat completion failed: {e}")
            return {
                "response": "",
                "error": str(e),
                "model": model_name,
                "response_time": 0,
                "success": False
            }
    
    def batch_generate(
        self,
        model_name: str,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate responses for multiple prompts"""
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            
            result = self.generate_response(
                model_name=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result["prompt_index"] = i
            results.append(result)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
        
        return results
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        try:
            response = self.session.post(
                f"{self.host}/api/show",
                json={"name": model_name},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get model info: {e}")
            return None
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model"""
        try:
            response = self.session.delete(
                f"{self.host}/api/delete",
                json={"name": model_name},
                timeout=30
            )
            response.raise_for_status()
            self.logger.info(f"Successfully deleted model {model_name}")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to delete model {model_name}: {e}")
            return False