"""
Main script for running DLIG attribution analysis.
"""

import gc
import torch
from models.model_manager import ModelManager, GradientEnabledModel
from attribution.dlig_attribution import DLIGAttribution
from attribution.hook_manager import HookManager
from utils.data_processor import DataProcessor
from config import MAX_NEW_TOKENS, GENERATION_STEPS


def main():
    """Main function to run DLIG attribution analysis."""
    # Initialize components
    model_manager = ModelManager()
    model, tokenizer = model_manager.load_model_and_tokenizer()
    
    # Initialize DLIG attribution
    dlig_attribution = DLIGAttribution(model, tokenizer)
    
    # Initialize hook manager
    hook_manager = HookManager(model)
    hook_handle = hook_manager.register_hook()
    
    # Initialize data processor
    data_processor = DataProcessor(tokenizer)
    
    try:
        # Prepare input
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        inputs = data_processor.prepare_messages_and_inputs(messages)
        input_ids = inputs.input_ids.to(model_manager.get_model_device())
        attention_mask = inputs.attention_mask.to(model_manager.get_model_device())
        

        # Print input information
        original_input_length, original_tokens = data_processor.print_input_info(input_ids)

        # Set original input length and relevant tokens for DLIG computation
        dlig_attribution.set_original_input_length(original_input_length)
        dlig_attribution.set_relevant_token_indices(original_tokens)  # This sets and prints the indices
    
        relevant_indices = dlig_attribution.get_relevant_token_indices()
        
        # Initialize gradient-enabled model wrapper
        grad_model = GradientEnabledModel(model)
        
        # Run generation with DLIG computation
        print("Starting diffusion generation with DLIG computation...")
        
        output = grad_model.diffusion_generate_with_grad(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            output_history=True,
            return_dict_in_generate=True,
            steps=GENERATION_STEPS,
            generation_logits_hook_func=dlig_attribution.generation_logits_hook_func
        )
        
        # Get DLIG scores
        dlig_scores = dlig_attribution.get_dlig_scores()
        
        # Analyze results - now only showing relevant tokens
        data_processor.analyze_dlig_results(dlig_scores, original_tokens, original_input_length, relevant_indices)        
        # Export to CSV - will only include relevant tokens
        if dlig_scores:
            csv_filepath = data_processor.export_to_csv(dlig_scores, input_ids, relevant_indices)
            print(f"Results exported to: {csv_filepath}")
        
        print("\nGeneration completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        hook_manager.remove_hook()
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
