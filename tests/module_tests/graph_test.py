
import torch
import flashinfer
from transformers import AutoConfig

from sdprofiler.request import FlashInferMetadata
from sdprofiler.models.qwen2 import Qwen2ForCausalLM




def build_graph(model, input_ids, position_ids, flashinfer_metadata, key_value_cache_blocks):
    device = flashinfer_metadata.input_ids_indptr.device
    batch_size = flashinfer_metadata.input_ids_lengths.shape[0]

    max_kv_indices_len_per_request = model.config.max_position_embeddings // model.config.kv_cache_block_size
    max_kv_indices_len = max_kv_indices_len_per_request * batch_size
    q_indptr_buffer = torch.empty(batch_size + 1, device=device).int()
    kv_indptr_buffer = torch.empty(batch_size + 1, device=device).int()
    kv_indices_buffer = torch.empty(max_kv_indices_len, device=device).int()
    kv_last_page_len_buffer = torch.empty(batch_size, device=device).int()
    
    prefill_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        prefill_workspace_buffer, "NHD", True, q_indptr_buffer, kv_indptr_buffer, kv_indices_buffer, kv_last_page_len_buffer
    )

    prefill_wrapper.plan(
        qo_indptr=flashinfer_metadata.input_ids_indptr,
        paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
        paged_kv_indices=flashinfer_metadata.paged_kv_indices,
        paged_kv_last_page_len=flashinfer_metadata.paged_kv_last_page_len,
        num_qo_heads=model.config.num_attention_heads,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim_qk=model.config.hidden_size // model.config.num_attention_heads,
        page_size=model.config.kv_cache_block_size,
        causal=True,
        pos_encoding_mode="NONE", # TODO: Check
        use_fp16_qk_reduction=False,
        q_data_type=model.config.torch_dtype,
        kv_data_type=model.config.past_key_values_dtype,
    )

    model.config.prefill_wrapper = prefill_wrapper

    input_tensors = (input_ids, position_ids)
    
    # warmup
    for i in range(10):
        output_tensor = model(
            input_ids=input_tensors[0], 
            position_ids=input_tensors[1], 
            flashinfer_metadata=flashinfer_metadata,
            past_key_values=key_value_cache_blocks
        )
    
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            output_tensor = model(
                input_ids=input_tensors[0], 
                position_ids=input_tensors[1], 
                flashinfer_metadata=flashinfer_metadata,
                past_key_values=key_value_cache_blocks
            )
    torch.cuda.synchronize()

    output_tensor.fill_(0)

    return graph, input_tensors, output_tensor, flashinfer_metadata, prefill_wrapper



if __name__ == "__main__":

    DEVICE = 'cuda:0'
    kv_cache_block_size = 16
    num_kv_cache_blocks = 172
    torch_dtype = torch.bfloat16

    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B-Instruct", cache_dir="/workspace/cache", use_safetensors=True)
    config.past_key_values_dtype = torch_dtype
    config.kv_cache_block_size = kv_cache_block_size
    model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch_dtype, config=config, cache_dir="/workspace/cache", use_safetensors=True)
    model.eval()
    model.to(DEVICE)

    prefill_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        prefill_workspace_buffer, "NHD", False
    )

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    past_key_values = [
        torch.randn(
            num_kv_cache_blocks, 
            2, 
            kv_cache_block_size, 
            num_kv_heads, 
            head_dim, 
            dtype=torch_dtype, 
            device=DEVICE
        ) for _ in range(num_layers)
    ]

    padded_past_key_values = [
        past_key_value.detach().clone() for past_key_value in past_key_values
    ]

    graph_past_key_values = [
        past_key_value.detach().clone() for past_key_value in past_key_values
    ]
    

    input_ids = torch.tensor([39814, 95456, 39814, 40], device=DEVICE, dtype=torch.int32)
    position_ids = torch.tensor([270, 269, 270, 339], device=DEVICE, dtype=torch.int32)

    flashinfer_metadata = FlashInferMetadata(
        input_ids_indptr = torch.tensor([0, 1, 2, 3, 4], device=DEVICE, dtype=torch.int32),
        input_ids_lengths = torch.tensor([1, 1, 1, 1], device=DEVICE, dtype=torch.int32),
        batch_indices = torch.tensor([0, 1, 2, 3], device=DEVICE, dtype=torch.int32),
        positions = torch.tensor([270, 269, 270, 339], device=DEVICE, dtype=torch.int32),
        paged_kv_indices = torch.arange(73, device=DEVICE, dtype=torch.int32),
        paged_kv_indptr = torch.tensor([0, 17, 34, 51, 73], device=DEVICE, dtype=torch.int32),
        paged_kv_last_page_len = torch.tensor([15, 14, 15, 4], device=DEVICE, dtype=torch.int32),
        paged_kv_indices_len = None,
        kv_cache_block_size=config.kv_cache_block_size,
        num_all_tokens=4,
        num_requests=4,
    )

    prefill_wrapper.plan(
        qo_indptr=flashinfer_metadata.input_ids_indptr,
        paged_kv_indptr=flashinfer_metadata.paged_kv_indptr,
        paged_kv_indices=flashinfer_metadata.paged_kv_indices,
        paged_kv_last_page_len=flashinfer_metadata.paged_kv_last_page_len,
        num_qo_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim_qk=config.hidden_size // config.num_attention_heads,
        page_size=kv_cache_block_size,
        causal=True,
        pos_encoding_mode="NONE", # TODO: Check
        use_fp16_qk_reduction=False,
        q_data_type=torch_dtype,
        kv_data_type=torch_dtype,
    )
    model.config.prefill_wrapper = prefill_wrapper
    torch.cuda.synchronize()

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            flashinfer_metadata=flashinfer_metadata,
            past_key_values=past_key_values
        )
    torch.cuda.synchronize()
    print(output[:,0])


    torch.cuda.synchronize()


    padded_input_ids = torch.tensor([39814, 95456, 39814, 40, 0, 0, 0, 0], device=DEVICE, dtype=torch.int32)
    padded_position_ids = torch.tensor([270, 269, 270, 339, 0, 0, 0, 0], device=DEVICE, dtype=torch.int32)
    
    # padded_flashinfer_metadata = FlashInferMetadata(
    #     input_ids_indptr=torch.tensor([0, 1, 2, 3, 7], device=DEVICE, dtype=torch.int32),
    #     input_ids_lengths=torch.tensor([1, 1, 1, 5], device=DEVICE, dtype=torch.int32),
    #     batch_indices=torch.tensor([0, 1, 2, 3, 3, 3, 3, 3], device=DEVICE, dtype=torch.int32),
    #     positions=torch.tensor([270, 269, 270, 339, 340, 341, 342, 343], device=DEVICE, dtype=torch.int32),
    #     paged_kv_indices=torch.empty(168, device=DEVICE, dtype=torch.int32),
    #     paged_kv_indptr=torch.tensor([0, 17, 34, 51, 73], device=DEVICE, dtype=torch.int32),
    #     paged_kv_last_page_len=torch.tensor([15, 14, 15, 4], device=DEVICE, dtype=torch.int32),
    #     paged_kv_indices_len=73,
    #     kv_cache_block_size=kv_cache_block_size,
    #     num_all_tokens=8,
    #     num_requests=4
    # )
    

    padded_flashinfer_metadata = FlashInferMetadata(
        input_ids_indptr=torch.tensor([0, 1, 2, 3, 4], device=DEVICE, dtype=torch.int32),
        input_ids_lengths=torch.tensor([1, 1, 1, 1], device=DEVICE, dtype=torch.int32),
        batch_indices=torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], device=DEVICE, dtype=torch.int32),
        positions=torch.tensor([270, 269, 270, 339, 0, 0, 0, 0], device=DEVICE, dtype=torch.int32),
        paged_kv_indices = torch.arange(168, device=DEVICE, dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 17, 34, 51, 73], device=DEVICE, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([15, 14, 15, 4], device=DEVICE, dtype=torch.int32),
        paged_kv_indices_len=73,
        kv_cache_block_size=config.kv_cache_block_size,
        num_all_tokens=4,
        num_requests=4
    )

    prefill_wrapper.plan(
        qo_indptr=padded_flashinfer_metadata.input_ids_indptr,
        paged_kv_indptr=padded_flashinfer_metadata.paged_kv_indptr,
        paged_kv_indices=padded_flashinfer_metadata.paged_kv_indices,
        paged_kv_last_page_len=padded_flashinfer_metadata.paged_kv_last_page_len,
        num_qo_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim_qk=config.hidden_size // config.num_attention_heads,
        page_size=kv_cache_block_size,
        causal=True,
        pos_encoding_mode="NONE", # TODO: Check
        use_fp16_qk_reduction=False,
        q_data_type=torch_dtype,
        kv_data_type=torch_dtype,
    )

    model.config.prefill_wrapper = prefill_wrapper
    torch.cuda.synchronize()
    with torch.no_grad():
        padded_output = model(
            input_ids=padded_input_ids,
            position_ids=padded_position_ids,
            flashinfer_metadata=padded_flashinfer_metadata,
            past_key_values=padded_past_key_values
        )
    torch.cuda.synchronize()
    print(padded_output[:,0])


    graph, graph_input_tensors, graph_output_tensor, graph_flashinfer_metadata, graph_prefill_wrapper = build_graph(
        model,
        input_ids = torch.randint(0, 10000, (padded_input_ids.shape), device=DEVICE, dtype=torch.int32),
        position_ids = torch.randint(0, 10000, (padded_position_ids.shape), device=DEVICE, dtype=torch.int32),
        flashinfer_metadata = padded_flashinfer_metadata,
        key_value_cache_blocks = graph_past_key_values
    )

    graph_input_tensors[0].copy_(padded_input_ids)
    graph_input_tensors[1].copy_(padded_position_ids)
    graph_flashinfer_metadata._copy_unsized(flashinfer_metadata)

    graph_prefill_wrapper.plan(
        qo_indptr=graph_flashinfer_metadata.input_ids_indptr,
        paged_kv_indptr=graph_flashinfer_metadata.paged_kv_indptr,
        paged_kv_indices=graph_flashinfer_metadata.paged_kv_indices,
        paged_kv_last_page_len=graph_flashinfer_metadata.paged_kv_last_page_len,
        num_qo_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim_qk=config.hidden_size // config.num_attention_heads,
        page_size=kv_cache_block_size,
        causal=True,
        pos_encoding_mode="NONE", # TODO: Check
        use_fp16_qk_reduction=False,
        q_data_type=torch_dtype,
        kv_data_type=torch_dtype,
    )

    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    print(graph_output_tensor[:,0])



#     graph output tensor: tensor([-18.8750,  -4.1562,  -3.1250,   7.7812, -17.5000, -17.5000, -17.5000,
#         -17.5000], device='cuda:0', dtype=torch.bfloat16)
# padded output tensor: tensor([-19.0000,  -4.0938,  -3.2344,   8.6875, -17.5000, -17.5000, -17.5000,
#         -17.5000], device='cuda:0', dtype=torch.bfloat16)