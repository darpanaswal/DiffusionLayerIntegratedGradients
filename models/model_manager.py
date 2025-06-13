
"""
Model loading and management utilities.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from config import MODEL_PATH, DEVICE_MAP, TORCH_DTYPE


class ModelManager:
    def __init__(self, model_path=MODEL_PATH, device_map=DEVICE_MAP, torch_dtype=TORCH_DTYPE):
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.model = None
        self.tokenizer = None
        
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            local_files_only=True
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        print("Model and tokenizer loaded successfully!")
        return self.model, self.tokenizer
    
    def get_model_device(self):
        """Get the device of the model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        return next(self.model.parameters()).device


class GradientEnabledModel:
    """Wrapper for enabling gradient computation during generation."""
    
    def __init__(self, model):
        self.model = model
        
    def diffusion_generate_with_grad(self, *args, **kwargs):
        """Generate with gradient computation enabled."""
        with torch.enable_grad():
            self.model.train()
            try:
                generation_config = self.model._prepare_generation_config(
                    kwargs.get('generation_config'), 
                    **{k: v for k, v in kwargs.items() if k != 'generation_config'}
                )
                
                generation_tokens_hook_func = kwargs.pop(
                    "generation_tokens_hook_func", 
                    lambda step, x, logits: x
                )
                generation_logits_hook_func = kwargs.pop(
                    "generation_logits_hook_func", 
                    lambda step, x, logits: logits
                )
                
                input_ids = args[0]
                attention_mask = kwargs.get("attention_mask")
                device = input_ids.device
                self.model._prepare_special_tokens(generation_config, device=device)
                
                input_ids_length = input_ids.shape[-1]
                has_default_max_length = (
                    kwargs.get("max_length") is None and 
                    generation_config.max_length is not None
                )
                
                generation_config = self.model._prepare_generated_length(
                    generation_config=generation_config,
                    has_default_max_length=has_default_max_length,
                    input_ids_length=input_ids_length,
                )
                
                self.model._validate_generated_length(
                    generation_config, input_ids_length, has_default_max_length
                )
                
                input_ids, attention_mask = self.model._expand_inputs_for_generation(
                    expand_size=generation_config.num_return_sequences,
                    input_ids=input_ids,
                    attention_mask=attention_mask 
                )
                
                result = self.model._sample(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    generation_tokens_hook_func=generation_tokens_hook_func,
                    generation_logits_hook_func=generation_logits_hook_func
                )
                return result
            finally:
                self.model.eval()
