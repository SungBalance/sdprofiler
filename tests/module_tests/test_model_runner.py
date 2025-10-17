import torch
import numpy as np
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


from sdprofiler.models.modeling_qwen2 import Qwen2ForCausalLM
from sdprofiler.model_runner import ModelRunner
from sdprofiler.registry import EngineRegistry
from sdprofiler.request import FlashInferMetadata
from sdprofiler.model_io_processor import ModelIOProcessor
from sdprofiler.config import ModelConfig, EngineConfig, ParallelConfig
from sdprofiler.utils.common import print_debug

def run_huggingface_model(tokenizer, hf_model, input_texts):
    # Tokenize batch of inputs for HuggingFace model
    hf_inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to("cuda")
    hf_input_ids = hf_inputs["input_ids"]
    hf_position_ids = torch.arange(1, hf_input_ids.size(-1) + 1).unsqueeze(0).expand(hf_input_ids.size(0), -1)

    # Get actual lengths before padding
    attention_mask = hf_inputs["attention_mask"]
    seq_lengths = attention_mask.sum(dim=1)


    print_debug("run_huggingface_model", 
        hf_input_ids=hf_input_ids,
        hf_position_ids=hf_position_ids,
        attention_mask=attention_mask,
        seq_lengths=seq_lengths
    )

    # Run HuggingFace model to get logits
    with torch.no_grad():
        hf_outputs = hf_model(**hf_inputs, output_hidden_states=True, output_attentions=True)
        hf_logits = hf_outputs.logits
        hf_hidden_states = hf_outputs.hidden_states[0]
        hf_attentions = hf_outputs.attentions[0]

    # Split logits based on actual sequence lengths
    split_hf_logits = []
    start_idx = 0
    for length in seq_lengths:
        length = length.item()
        split_tensor = hf_logits[start_idx, :length, :].squeeze(0)
        split_hf_logits.append(split_tensor)
        start_idx += 1
        
    return split_hf_logits


def run_custom_model(
        model_io_processor: ModelIOProcessor,
        model_runner: ModelRunner,
        tokenizer: AutoTokenizer,
        input_texts: List[str]
    ):
    def run_tokenizer(input_texts):
        input_ids_list = [tokenizer(input_text, return_tensors="np").input_ids[0] for input_text in input_texts]
        position_ids_list = [np.arange(1, len(input_ids) + 1) for input_ids in input_ids_list]
        return input_ids_list, position_ids_list
    
    def build_input_tensors(input_ids_list, position_ids_list):
        # Allocate paged KV cache
        paged_kv_indices_list = []
        paged_kv_last_page_len_list = []
        
        # Track total pages for offset
        total_pages = 0
        for input_ids in input_ids_list:
            page_size = model_io_processor.kv_cache_block_size
            seq_len = len(input_ids)
            num_pages = (seq_len + page_size - 1) // page_size
            
            # Create sequential page indices with offset
            paged_kv_indices = np.arange(total_pages, total_pages + num_pages, dtype=np.int32)
            total_pages += num_pages
            
            # Calculate the length of the last page
            last_page_len = seq_len % page_size if seq_len % page_size != 0 else page_size
            paged_kv_last_page_len = np.array([last_page_len], dtype=np.int32)
            
            paged_kv_indices_list.append(paged_kv_indices)
            paged_kv_last_page_len_list.append(paged_kv_last_page_len)

        # Build model inputs using model_io_processor - adjusted to match method signature
        input_ids_tensor, position_ids_tensor, flashinfer_metadata = model_io_processor._build_flashinfer_model_inputs(
            input_ids_list=input_ids_list,
            position_ids_list=position_ids_list,
            paged_kv_indices_list=paged_kv_indices_list,
            paged_kv_last_page_len_list=paged_kv_last_page_len_list
        )
        
        print_debug("build_input_tensors: after", 
            batch_size=flashinfer_metadata.input_ids_indptr.shape[0]-1,
            input_ids_tensor=input_ids_tensor,
            position_ids_tensor=position_ids_tensor,
            input_ids_indptr=flashinfer_metadata.input_ids_indptr,
            input_ids_lengths=flashinfer_metadata.input_ids_lengths,
            paged_kv_indices=flashinfer_metadata.paged_kv_indices,
            paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
            paged_kv_last_page_len=flashinfer_metadata.paged_kv_last_page_len
        )
    
        return input_ids_tensor, position_ids_tensor, flashinfer_metadata
        
    input_ids_list, position_ids_list = run_tokenizer(input_texts)
    input_ids_tensor, position_ids_tensor, flashinfer_metadata = build_input_tensors(input_ids_list, position_ids_list)

    # Run custom ModelRunner to get logits
    with torch.no_grad():
        custom_logits = model_runner.run_model(
            input_ids_tensor=input_ids_tensor,
            position_ids_tensor=position_ids_tensor,
            flashinfer_metadata=flashinfer_metadata,
            num_logits_to_keep=None
        )
    split_custom_logits = []
    input_ids_indptr_cpu = flashinfer_metadata.input_ids_indptr.cpu().numpy()
    for i in range(len(input_ids_indptr_cpu) - 1):
        start_idx = input_ids_indptr_cpu[i]
        end_idx = input_ids_indptr_cpu[i + 1]
        split_custom_logits.append(custom_logits[start_idx:end_idx])
    return split_custom_logits


def test_model_runner_vs_huggingface():
    # Initialize models
    model_name_or_path = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    hf_model = Qwen2ForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
    hf_model.cuda()
    hf_model.eval()



    model_config = ModelConfig(model_name_or_path=model_name_or_path)
    engine_config = EngineConfig()
    parallel_config = ParallelConfig()
    registry = EngineRegistry(model_config, engine_config, parallel_config)
    model_io_processor = ModelIOProcessor(registry)
    model_runner = ModelRunner(registry)
    model_runner.model.eval()

    # Sample input texts
    input_texts = [
        "Once upon a time",
        # "In a galaxy far far away"  # Added another example for batching
    ]

    # Run HuggingFace model with batched inputs
    hf_logits_list = run_huggingface_model(tokenizer, hf_model, input_texts)
    
    # Run custom model with batched inputs
    custom_logits_list = run_custom_model(
        model_io_processor,
        model_runner,
        tokenizer,
        input_texts,
    )

    for hf_logits, custom_logits in zip(hf_logits_list, custom_logits_list):
        print_debug("compare_logits", 
            hf_logits_shape=hf_logits.shape,
            custom_logits_shape=custom_logits.shape,
            hf_logits_device=hf_logits.device,
            custom_logits_device=custom_logits.device,
        )
        if not torch.allclose(hf_logits, custom_logits, atol=1e-5):
            assert False, f"Logits do not match. Max difference: {torch.abs(hf_logits - custom_logits).max()}\nMax Value: hf={hf_logits.max()}, custom={custom_logits.max()}"

    print("Test Passed: Custom ModelRunner logits match HuggingFace logits.")



if __name__ == "__main__":
    test_model_runner_vs_huggingface()
