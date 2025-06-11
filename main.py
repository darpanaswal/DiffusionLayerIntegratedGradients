import gc
import os
import csv
import torch
import traceback
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# del model
# del tokenizer
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()


model_path = "Dream-org/Dream-v0-Instruct-7B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

layer_name = 'model.embed_tokens'  # or any Transformer layer like 'model.layers.10'
target_layer = dict()

def capture_activations(module, inp, out):
    out = out if isinstance(out, torch.Tensor) else out[0]
    out = out.detach().clone().requires_grad_(True)
    target_layer['output'] = out
    print(f"[DEBUG] Hooked layer output shape: {out.shape}")

hooked_layer = eval(f"model.{layer_name}")
hook_handle = hooked_layer.register_forward_hook(capture_activations)

dlig_scores = []
original_input_length = None
integration_steps = 20  # Number of integration steps (m in the formula)

def create_baseline_input(x_t, mask_token_id, original_length):
    # x_t: [1, curr_len]
    baseline = x_t.clone()
    baseline[:, original_length:] = mask_token_id  # mask everything after the original prompt
    return baseline

def compute_dlig_at_timestep(step, x_t, logits, mask_token_id, original_length, attention_mask):
    print(f"[DLIG DEBUG] step={step} x_t.shape={x_t.shape} attention_mask.shape={attention_mask.shape}")
    try:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        curr_len = x_t.shape[1]  # current full sequence length, including generated tokens
        x_prime_t = create_baseline_input(x_t, mask_token_id, original_length)

        with torch.no_grad():
            model(inputs_embeds=model.model.embed_tokens(x_t), attention_mask=attention_mask)
            h_real = target_layer['output'].detach()
        
            model(inputs_embeds=model.model.embed_tokens(x_prime_t), attention_mask=attention_mask)
            h_baseline = target_layer['output'].detach()

        accumulated_gradients = None

        model.train()  # enable grads

        for k in range(1, integration_steps + 1):
            alpha_k = k / integration_steps
            interpolated_activations = h_baseline + alpha_k * (h_real - h_baseline)
            interpolated_activations = interpolated_activations.detach().clone().requires_grad_(True)
        
            def replace_hook(_module, _input):
                return interpolated_activations
        
            hook_handle.remove()
            dummy_hook = hooked_layer.register_forward_hook(lambda m, i, o: interpolated_activations)
        
            model.zero_grad(set_to_none=True)
            outputs = model(inputs_embeds=model.model.embed_tokens(x_t), attention_mask=attention_mask)
            logits = outputs.logits
            target_logit = logits[0, -1, :].max()
            target_logit.backward(retain_graph=True)
        
            grad = interpolated_activations.grad.clone()
            accumulated_gradients = grad if accumulated_gradients is None else accumulated_gradients + grad
        
            dummy_hook.remove()
            hook_handle = hooked_layer.register_forward_hook(capture_activations)

        model.eval()

        if accumulated_gradients is not None:
            avg_gradients = accumulated_gradients / integration_steps
            activation_diff = h_real - h_baseline
            dlig_raw = activation_diff * avg_gradients

            # Only for reporting: focus on the prompt tokens
            input_dlig = dlig_raw[:, :original_length, :].detach().cpu()
            token_dlig_scores = input_dlig.norm(dim=-1)
            return {
                'step': step,
                'token_scores': token_dlig_scores,
                'full_dlig': input_dlig,
                'activation_diff_norm': activation_diff[:, :original_length, :].norm(dim=-1).detach().cpu()
            }
    except Exception as e:
        import traceback
        print(f"DLIG computation failed at step {step}: {e}")
        traceback.print_exc()
        return None

def generation_logits_hook_func(step, x, logits):
    global original_input_length
    # print(f"[GEN HOOK DEBUG] step={step} x.shape={x.shape} attention_mask.shape={attention_mask.shape}")
    if step is not None and original_input_length is not None:
        mask_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # Compute DLIG for this time-step
        attention_mask = torch.ones_like(x)  # shape [1, curr_len], 1 for all tokens
        print(f"[GEN HOOK DEBUG] step={step} x.shape={x.shape} attention_mask.shape={attention_mask.shape}")
        dlig_result = compute_dlig_at_timestep(step, x, logits, mask_token_id, original_input_length, attention_mask)

        
        if dlig_result is not None:
            dlig_scores.append(dlig_result)
            print(f"\033[92m[Progress]\033[0m Step {step}: DLIG computed. Token scores shape: {dlig_result['token_scores'].shape}")
        else:
            print(f"\033[91m[Progress]\033[0m Step {step}: DLIG computation failed.")

    else:
        print(f"\033[93m[Progress]\033[0m Step {step}: Skipped attribution (insufficient info).")
    return logits

messages = [
    {"role": "user", "content": "What is the capital of France?"}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)

# Store the original input length
original_input_length = input_ids.shape[1]
print(f"Original input length: {original_input_length}")

# Decode and print the original input for reference
original_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print(f"Original input tokens: {original_tokens}")

class GradientEnabledModel:
    def __init__(self, model):
        self.model = model
        
    def diffusion_generate_with_grad(self, *args, **kwargs):
        with torch.enable_grad():
            self.model.train()
            try:
                generation_config = self.model._prepare_generation_config(
                    kwargs.get('generation_config'), **{k: v for k, v in kwargs.items() if k != 'generation_config'}
                )
                
                generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
                generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)
                
                input_ids = args[0]
                attention_mask = kwargs.get("attention_mask")
                device = input_ids.device
                self.model._prepare_special_tokens(generation_config, device=device)
                
                input_ids_length = input_ids.shape[-1]
                has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
                generation_config = self.model._prepare_generated_length(
                    generation_config=generation_config,
                    has_default_max_length=has_default_max_length,
                    input_ids_length=input_ids_length,
                )
                
                self.model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
                
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

# Run the generation with DLIG computation
print("Starting diffusion generation with DLIG computation...")
grad_model = GradientEnabledModel(model)

output = grad_model.diffusion_generate_with_grad(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=64,
    output_history=True,
    return_dict_in_generate=True,
    steps=32,
    generation_logits_hook_func=generation_logits_hook_func
)

# Clean up
hook_handle.remove()

# Analyze results
print(f"\nDLIG Analysis Complete!")
print(f"Total time-steps with DLIG computed: {len(dlig_scores)}")

if dlig_scores:
    print(f"Shape of token scores per step: {dlig_scores[0]['token_scores'].shape}")
    print(f"Shape of full DLIG per step: {dlig_scores[0]['full_dlig'].shape}")
    
    # Show attribution evolution over time-steps
    print("\nAttribution scores evolution (first 5 steps):")
    for i, dlig_data in enumerate(dlig_scores[:5]):
        step = dlig_data['step']
        token_scores = dlig_data['token_scores'][0]  # Remove batch dimension
        print(f"Step {step}: {token_scores.tolist()}")
    
    # Show token-wise attribution across all time-steps
    print(f"\nToken-wise attribution across {len(dlig_scores)} time-steps:")
    all_token_scores = torch.stack([d['token_scores'][0] for d in dlig_scores])  # [steps, tokens]
    print(f"All scores shape: {all_token_scores.shape}")
    
    for token_idx in range(min(5, original_input_length)):  # Show first 5 tokens
        token_attribution_over_time = all_token_scores[:, token_idx]
        print(f"Token {token_idx} ('{original_tokens[token_idx]}'): {token_attribution_over_time.tolist()}")

print("\nGeneration completed successfully!")

# Directory and filename
csv_dir = "./attribution_csv"
os.makedirs(csv_dir, exist_ok=True)
csv_filename = os.path.join(csv_dir, "dlig_attribution_scores.csv")

# Get tokens once (they should be same for each step)
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

fieldnames = ['step', 'token_idx', 'token', 'score', 'flattened_score']

with open(csv_filename, "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for dlig in dlig_scores:
        step = dlig['step']
        # full_dlig: [1, input_len, hidden_dim], remove batch dim
        full_dlig = dlig['full_dlig'][0]  # shape: (input_len, hidden_dim)
        for token_idx, token in enumerate(tokens):
            raw_score_vec = full_dlig[token_idx].tolist()
            flattened_score = float(torch.tensor(raw_score_vec).mean())
            writer.writerow({
                'step': step,
                'token_idx': token_idx,
                'token': token,
                'score': raw_score_vec,               # stored as list in CSV; can be loaded as string-eval later
                'flattened_score': flattened_score
            })

print(f"Attribution scores for each step/token written to: {csv_filename}")
