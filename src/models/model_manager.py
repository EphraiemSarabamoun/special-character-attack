"""Model manager for coordinating multiple model interactions"""

import yaml
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import concurrent.futures
from dataclasses import dataclass

from .ollama_client import OllamaClient


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    size_category: str  # small, medium, large
    parameter_count: Optional[str] = None
    architecture: Optional[str] = None
    context_length: Optional[int] = None
    available: bool = False


class ModelManager:
    """Manages multiple models for testing"""
    
    def __init__(self, config_path: Optional[str] = None, ollama_host: str = "http://localhost:11434"):
        self.logger = logging.getLogger(__name__)
        self.ollama_client = OllamaClient(host=ollama_host)
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        self.config = self._load_config(config_path)
        self.models = self._initialize_models()
        
        # Track model availability
        self._check_model_availability()
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            # Return default configuration
            return {
                "models": {
                    "small": ["phi3:3.8b", "gemma2:2b", "mistral:7b"],
                    "medium": ["llama3.2:11b", "phi3:14b", "gemma2:7b"],
                    "large": ["llama3.1:70b", "mixtral:8x7b", "qwen2.5:72b"]
                }
            }
    
    def _initialize_models(self) -> Dict[str, ModelInfo]:
        """Initialize model information from config"""
        models = {}
        
        for size_category, model_list in self.config.get("models", {}).items():
            for model_name in model_list:
                models[model_name] = ModelInfo(
                    name=model_name,
                    size_category=size_category,
                    parameter_count=self._extract_parameter_count(model_name),
                    architecture=self._infer_architecture(model_name)
                )
        
        return models
    
    def _extract_parameter_count(self, model_name: str) -> Optional[str]:
        """Extract parameter count from model name"""
        import re
        
        # Match patterns like 7b, 3.8b, 70b, 8x7b, etc.
        patterns = [
            r'(\d+\.?\d*[bB])',  # 7b, 3.8b, 70b
            r'(\d+x\d+[bB])',    # 8x7b
            r':(\d+\.?\d*[bB])', # :7b
            r':(\d+x\d+[bB])'    # :8x7b
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_name)
            if match:
                return match.group(1).replace(':', '')
        
        return None
    
    def _infer_architecture(self, model_name: str) -> Optional[str]:
        """Infer model architecture from name"""
        name_lower = model_name.lower()
        
        if 'llama' in name_lower:
            return 'Llama'
        elif 'mistral' in name_lower or 'mixtral' in name_lower:
            return 'Mistral'
        elif 'gemma' in name_lower:
            return 'Gemma'
        elif 'phi' in name_lower:
            return 'Phi'
        elif 'qwen' in name_lower:
            return 'Qwen'
        elif 'deepseek' in name_lower:
            return 'DeepSeek'
        elif 'gpt-oss' in name_lower or 'gptoss' in name_lower:
            return 'GPT-OSS'
        
        return 'Unknown'
    
    def _check_model_availability(self) -> None:
        """Check which models are available"""
        available_models = self.ollama_client.list_models()
        available_names = {model['name'] for model in available_models}
        
        for model_name, model_info in self.models.items():
            # Check for exact match or prefix match
            model_info.available = any(
                available_name.startswith(model_name.split(':')[0]) 
                for available_name in available_names
            )
            
            if model_info.available:
                self.logger.info(f"Model {model_name} is available")
            else:
                self.logger.warning(f"Model {model_name} is not available")
    
    def get_available_models(self, size_category: Optional[str] = None) -> List[ModelInfo]:
        """Get list of available models, optionally filtered by size category"""
        available = [model for model in self.models.values() if model.available]
        
        if size_category:
            available = [model for model in available if model.size_category == size_category]
        
        return available
    
    def get_unavailable_models(self) -> List[ModelInfo]:
        """Get list of unavailable models"""
        return [model for model in self.models.values() if not model.available]
    
    def pull_missing_models(self, size_categories: Optional[List[str]] = None) -> Dict[str, bool]:
        """Pull missing models for specified size categories"""
        if size_categories is None:
            size_categories = ["small", "medium"]  # Don't automatically pull large models
        
        results = {}
        
        for model_name, model_info in self.models.items():
            if not model_info.available and model_info.size_category in size_categories:
                self.logger.info(f"Pulling model {model_name}...")
                success = self.ollama_client.pull_model(model_name)
                results[model_name] = success
                
                if success:
                    model_info.available = True
                    self.logger.info(f"Successfully pulled {model_name}")
                else:
                    self.logger.error(f"Failed to pull {model_name}")
        
        return results
    
    def test_single_prompt(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Test a single prompt across multiple models"""
        
        if models is None:
            models = [m.name for m in self.get_available_models()]
        
        results = {}
        
        for model_name in models:
            if model_name not in self.models:
                self.logger.warning(f"Model {model_name} not in configuration")
                continue
                
            if not self.models[model_name].available:
                self.logger.warning(f"Model {model_name} not available")
                continue
            
            self.logger.info(f"Testing prompt with model {model_name}")
            
            result = self.ollama_client.generate_response(
                model_name=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Add model metadata
            result["model_info"] = {
                "size_category": self.models[model_name].size_category,
                "parameter_count": self.models[model_name].parameter_count,
                "architecture": self.models[model_name].architecture
            }
            
            results[model_name] = result
        
        return results
    
    def test_prompts_batch(
        self,
        prompts: List[str],
        models: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_workers: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Test multiple prompts across models with parallel processing"""
        
        if models is None:
            models = [m.name for m in self.get_available_models()]
        
        # Filter to available models only
        available_models = [m for m in models if m in self.models and self.models[m].available]
        
        results = {model: [] for model in available_models}
        
        def test_model_prompts(model_name: str) -> Tuple[str, List[Dict[str, Any]]]:
            """Test all prompts for a single model"""
            model_results = []
            
            for i, prompt in enumerate(prompts):
                self.logger.info(f"Model {model_name}: Testing prompt {i+1}/{len(prompts)}")
                
                result = self.ollama_client.generate_response(
                    model_name=model_name,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Add metadata
                result["prompt_index"] = i
                result["model_info"] = {
                    "size_category": self.models[model_name].size_category,
                    "parameter_count": self.models[model_name].parameter_count,
                    "architecture": self.models[model_name].architecture
                }
                
                model_results.append(result)
                
                # Small delay between requests
                time.sleep(0.1)
            
            return model_name, model_results
        
        # Execute in parallel (limited concurrency to avoid overwhelming Ollama)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(available_models))) as executor:
            future_to_model = {
                executor.submit(test_model_prompts, model): model 
                for model in available_models
            }
            
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    returned_model, model_results = future.result()
                    results[returned_model] = model_results
                    self.logger.info(f"Completed testing for model {returned_model}")
                except Exception as e:
                    self.logger.error(f"Model {model_name} failed: {e}")
                    results[model_name] = []
        
        return results
    
    def compare_model_responses(
        self,
        prompt: str,
        models: Optional[List[str]] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Compare responses from different models for analysis"""
        
        responses = self.test_single_prompt(
            prompt=prompt,
            models=models,
            temperature=temperature
        )
        
        # Analyze responses
        analysis = {
            "prompt": prompt,
            "timestamp": time.time(),
            "responses": responses,
            "analysis": {
                "response_lengths": {model: len(resp["response"]) for model, resp in responses.items()},
                "response_times": {model: resp["response_time"] for model, resp in responses.items()},
                "success_rates": {model: resp["success"] for model, resp in responses.items()},
                "model_architectures": {
                    model: resp["model_info"]["architecture"] 
                    for model, resp in responses.items() 
                    if "model_info" in resp
                }
            }
        }
        
        return analysis
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about configured and available models"""
        total_models = len(self.models)
        available_models = len(self.get_available_models())
        unavailable_models = total_models - available_models
        
        by_size = {}
        for size_cat in ["small", "medium", "large"]:
            models_in_cat = [m for m in self.models.values() if m.size_category == size_cat]
            available_in_cat = [m for m in models_in_cat if m.available]
            by_size[size_cat] = {
                "total": len(models_in_cat),
                "available": len(available_in_cat),
                "models": [m.name for m in models_in_cat]
            }
        
        by_architecture = {}
        for model in self.models.values():
            arch = model.architecture or "Unknown"
            if arch not in by_architecture:
                by_architecture[arch] = {"total": 0, "available": 0, "models": []}
            
            by_architecture[arch]["total"] += 1
            by_architecture[arch]["models"].append(model.name)
            if model.available:
                by_architecture[arch]["available"] += 1
        
        return {
            "total_models": total_models,
            "available_models": available_models,
            "unavailable_models": unavailable_models,
            "availability_rate": available_models / total_models if total_models > 0 else 0,
            "by_size_category": by_size,
            "by_architecture": by_architecture
        }