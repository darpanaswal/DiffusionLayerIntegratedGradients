"""
DLIG (Diffusion Language Integrated Gradients) attribution implementation.
"""

import torch
import traceback
from config import INTEGRATION_STEPS


class DLIGAttribution:
    def __init__(self, model, tokenizer, integration_steps=INTEGRATION_STEPS):
        self.model = model
        self.tokenizer = tokenizer
        self.integration_steps = integration_steps
        self.dlig_scores = []
        self.original_input_length = None
        self.relevant_token_indices = []  # To store indices of relevant tokens
        
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

            curr_len = x_t.shape[1]  # current full sequence length, including generated tokens
            x_prime_t = self.create_baseline_input(x_t, mask_token_id, original_length)

            real_embeds = self.model.model.embed_tokens(x_t)         # [1, curr_len, dim]
            baseline_embeds = self.model.model.embed_tokens(x_prime_t)  # [1, curr_len, dim]

            accumulated_gradients = torch.zeros_like(real_embeds)

            self.model.train()  # enable grads

            for k in range(1, self.integration_steps + 1):
                alpha_k = k / self.integration_steps
                
                # Fresh computation each time (detach from previous graph explicitly)
                interpolated_embeds = (
                    baseline_embeds + alpha_k * (real_embeds - baseline_embeds)
                ).detach().clone()
                interpolated_embeds.requires_grad_(True)

                self.model.zero_grad(set_to_none=True)
                outputs = self.model(inputs_embeds=interpolated_embeds, attention_mask=attention_mask)
                logits = outputs.logits

                target_logit = logits[0, -1, :].max()

                # Retain graph explicitly to avoid RuntimeError
                target_logit.backward(retain_graph=False)

                if interpolated_embeds.grad is not None:
                    accumulated_gradients += interpolated_embeds.grad.clone()
                else:
                    print(f"[WARN] Gradients not found at step {k}")

            avg_gradients = accumulated_gradients / self.integration_steps
            activation_diff = real_embeds - baseline_embeds
            activation_diff_norm = activation_diff.norm()
            print(f"Activation diff norm: {activation_diff_norm}")
            
            dlig_raw = activation_diff * avg_gradients

            # Only select relevant tokens for the final scores
            if not self.relevant_token_indices:
                print("[WARN] No relevant token indices set, using all tokens")
                input_dlig = dlig_raw[:, :original_length, :].detach().cpu()
                token_dlig_scores = input_dlig.norm(dim=-1)
            else:
                input_dlig = dlig_raw[:, self.relevant_token_indices, :].detach().cpu()
                token_dlig_scores = input_dlig.norm(dim=-1)
            
            return {
                'step': step,
                'token_scores': token_dlig_scores,
                'full_dlig': input_dlig,
                'activation_diff_norm': activation_diff[:, :original_length, :].norm(dim=-1).detach().cpu()
            }
            
        except Exception as e:
            print(f"DLIG computation failed at step {step}: {e}")
            traceback.print_exc()
            return None
    
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
