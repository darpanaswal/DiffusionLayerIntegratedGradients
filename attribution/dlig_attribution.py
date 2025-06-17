"""
DLIG (Diffusion Language Integrated Gradients) attribution implementation.
"""

import torch
import traceback
from config import INTEGRATION_STEPS
from utils.layer_utils import resolve_layer_path


class DLIGAttribution:
    def __init__(self, model, tokenizer, hook_manager, integration_steps=INTEGRATION_STEPS):
        self.model = model
        self.tokenizer = tokenizer
        self.integration_steps = integration_steps
        self.dlig_scores = []
        self.original_input_length = None
        self.relevant_token_indices = []
        self.target_output_ids = None
        self.hook_manager = hook_manager
    
    def set_target_output(self, target_text):
        """Set a specific target output for attribution calculation."""
        if target_text is None:
            self.target_output_ids = None
            print("Target output cleared - using variable output mode")
        else:
            self.target_output_ids = self.tokenizer.encode(
                target_text, 
                return_tensors="pt"
            ).to(self.model.device)
            print(f"Target output set: {self.tokenizer.decode(self.target_output_ids[0])}")
            print(f"Target token IDs: {self.target_output_ids.tolist()}")

    def identify_relevant_tokens(self, input_tokens):
        """Identify which tokens are relevant for attribution (user input)."""
        relevant_indices = []
        in_user_section = False
        user_started = False
        
        for i, token in enumerate(input_tokens):
            if token == '<|im_start|>' and i+1 < len(input_tokens) and input_tokens[i+1] == 'user':
                in_user_section = True
                user_started = True
                continue  # Skip the <|im_start|> and 'user' tokens
            elif token == '<|im_end|>':
                in_user_section = False
                continue
            elif in_user_section and user_started:
                # Only include tokens after we've seen the user section start
                relevant_indices.append(i)
        
        return relevant_indices
    
    def create_baseline_input(self, x_t, mask_token_id, original_length):
        """Create baseline input by masking original tokens."""
        baseline = x_t.clone()
        # Mask all original tokens, creating meaningful differences
        baseline[:, :original_length] = mask_token_id
        return baseline
    
    def compute_dlig_at_timestep(self, step, x_t, logits, mask_token_id, original_length, attention_mask):
        """Compute DLIG attribution for a given timestep."""
        print(f"[DLIG DEBUG] step={step} x_t.shape={x_t.shape} attention_mask.shape={attention_mask.shape}")
        
        try:
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()

            # curr_len = x_t.shape[1]  # current full sequence length, including generated tokens
            x_prime_t = self.create_baseline_input(x_t, mask_token_id, original_length)

            # First forward pass (real input)
            with torch.no_grad():
                self.hook_manager.real_activations = None  # Reset before capture
                _ = self.model(input_ids=x_t, attention_mask=attention_mask)
                real_activations = self.hook_manager.real_activations
        
            # Second forward pass (baseline input)
            with torch.no_grad():
                # Use a temporary hook for baseline
                baseline_hook = self.hook_manager.register_baseline_hook()
                try:
                    _ = self.model(input_ids=x_prime_t, attention_mask=attention_mask)
                    baseline_activations = self.hook_manager.baseline_activations
                finally:
                    baseline_hook.remove()

            # Ensure we have valid activations
            if real_activations is None or baseline_activations is None:
                raise ValueError(f"Failed to capture layer activations. Real: {real_activations is None}, Baseline: {baseline_activations is None}")

            accumulated_gradients = torch.zeros_like(real_activations)
            activation_diff = real_activations - baseline_activations
            
            self.model.train()  # enable grads

            for k in range(1, self.integration_steps + 1):
                alpha_k = k / self.integration_steps
                
                # Fresh computation each time
                with torch.set_grad_enabled(True):
                    interpolated_activations = (
                        baseline_activations + alpha_k * activation_diff
                    ).requires_grad_(True)

                    self.model.zero_grad()

                    # This requires modifying your model to accept hidden states as input
                    output = self._forward_from_layer(
                        interpolated_activations, 
                        self.hook_manager.layer_name,
                        attention_mask
                    )
                    
                    # Get target logit
                    if self.target_output_ids is not None:
                        target_logit = self._get_target_logit(output, step)
                    else:
                        target_logit = output.logits[0, -1, :].mean()

                    # Compute gradients
                    gradients = torch.autograd.grad(
                        outputs=target_logit,
                        inputs=interpolated_activations,
                        # retain_graph=False,
                        create_graph=False
                    )[0]
                    
                    if gradients is not None:
                        accumulated_gradients += gradients.detach()
                    else:
                        print(f"[WARN] Gradients not found at step {k}")

            avg_gradients = accumulated_gradients / self.integration_steps
            activation_diff_norm = activation_diff.norm()
            print(f"Activation diff norm: {activation_diff_norm}")
            
            dlig_raw = activation_diff * avg_gradients

            # Process and return results
            return self._process_results(dlig_raw, original_length, step, activation_diff)

        except Exception as e:
            print(f"DLIG computation failed: {e}")
            traceback.print_exc()
            return None

    def _forward_from_layer(self, activations, layer_name, attention_mask):
        """Continue forward pass from the given layer activations."""
        try:
            # Handle different model architectures
            model = self.model
            
            # Special case for embedding layer
            if layer_name == 'model.embed_tokens':
                # If we're starting from embeddings, we need to go through the entire model
                # Create fake input_ids (they won't be used since we're providing hidden_states)
                input_ids = torch.zeros((1, activations.shape[1]), 
                                    dtype=torch.long, 
                                    device=activations.device)
                
                # Forward through the rest of the model
                outputs = model(attention_mask=attention_mask, inputs_embeds=activations)
                return outputs
            
            # For other layers, we need to navigate the model structure
            curr = resolve_layer_path(model, layer_name)
            
            # Run remaining layers
            x = activations
            position_ids = torch.arange(x.shape[1], dtype=torch.long, device=x.device).unsqueeze(0)
            if hasattr(curr, 'forward'):
                # If it's a module, we need to call it properly
                x = curr(x, attention_mask=attention_mask, position_ids=position_ids)
                if isinstance(x, tuple):
                    x = x[0]  # Typically the first element is hidden states
                
                # Continue with the rest of the model
                # This part is model-specific - you'll need to adjust based on your architecture
                # Here's a generic approach that may need adjustment:
                if hasattr(model, 'layers_after'):
                    for layer in model.layers_after:
                        x = layer(x, attention_mask=attention_mask)
                        if isinstance(x, tuple):
                            x = x[0]
                
                return model(inputs_embeds=x, attention_mask=attention_mask)  # Final processing
            
            raise NotImplementedError("Could not determine how to continue forward pass")
        
        except Exception as e:
            print(f"Error in _forward_from_layer: {e}")
            raise
    
    def generation_logits_hook_func(self, step, x, logits):
        """Hook function for computing DLIG during generation."""
        if step is not None and self.original_input_length is not None:
            mask_token_id = (
                self.tokenizer.pad_token_id 
                if self.tokenizer.pad_token_id is not None 
                else self.tokenizer.eos_token_id
            )
            
            # Compute DLIG for this time-step
            attention_mask = torch.ones_like(x)  # shape [1, curr_len], 1 for all tokens
            print(f"[GEN HOOK DEBUG] step={step} x.shape={x.shape} attention_mask.shape={attention_mask.shape}")
            
            dlig_result = self.compute_dlig_at_timestep(
                step, x, logits, mask_token_id, self.original_input_length, attention_mask
            )
            
            if dlig_result is not None:
                self.dlig_scores.append(dlig_result)
                print(f"\033[92m[Progress]\033[0m Step {step}: DLIG computed. "
                      f"Token scores shape: {dlig_result['token_scores'].shape}")
            else:
                print(f"\033[91m[Progress]\033[0m Step {step}: DLIG computation failed.")
        else:
            print(f"\033[93m[Progress]\033[0m Step {step}: Skipped attribution (insufficient info).")
            
        return logits

    def _get_target_logit(self, output, step):
        """Get target logit based on current step."""
        if step >= self.target_output_ids.shape[1]:
            target_token_id = self.target_output_ids[0, -1]
        else:
            target_token_id = self.target_output_ids[0, step]
        return output.logits[0, -1, target_token_id]

    def _process_results(self, dlig_raw, original_length, step, activation_diff):
        """Process raw DLIG results into final scores."""
        if self.relevant_token_indices:
            input_dlig = dlig_raw[:, self.relevant_token_indices, :].detach().cpu()
            token_scores = input_dlig.norm(dim=-1)
        else:
            input_dlig = dlig_raw[:, :original_length, :].detach().cpu()
            token_scores = input_dlig.norm(dim=-1)
        
        return {
            'step': step,
            'token_scores': token_scores,
            'full_dlig': input_dlig,
            'activation_diff_norm': activation_diff[:, :original_length, :].norm(dim=-1).detach().cpu()
        }
    
    def set_original_input_length(self, length):
        """Set the original input length."""
        self.original_input_length = length

    def get_relevant_token_indices(self):
        return self.relevant_token_indices
    
    def set_relevant_token_indices(self, input_tokens):
        """Set indices of relevant tokens for attribution."""
        self.relevant_token_indices = self.identify_relevant_tokens(input_tokens)
        print(f"Relevant token indices: {self.relevant_token_indices}")
        print(f"Relevant tokens: {[input_tokens[i] for i in self.relevant_token_indices]}")
    
    def get_dlig_scores(self):
        """Get computed DLIG scores."""
        return self.dlig_scores
    
    def reset_scores(self):
        """Reset stored DLIG scores."""
        self.dlig_scores = []
        self.relevant_token_indices = []
