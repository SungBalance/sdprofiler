import torch
import numpy as np
import flashinfer


# model_args
num_qo_heads = 64
num_kv_heads = 16
head_dim = 128
device = "cuda:0"

# cache_args
max_num_pages = 48
page_size = 4
paged_kv_cache = torch.zeros(max_num_pages, 2, page_size, num_kv_heads, head_dim).half().to(device)

# allocate 128MB workspace buffer
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)

# input_args
batch_size = 4
num_tokens = [1, 2, 3, 4]
num_all_tokens = sum(num_tokens) # 100

k_append = torch.ones(num_all_tokens, num_kv_heads, head_dim).half().to(device)
v_append = torch.ones(num_all_tokens, num_kv_heads, head_dim).half().to(device)
q = torch.ones(num_all_tokens, num_qo_heads, head_dim).half().to(device)

for i in range(num_all_tokens):
    k_append[i,] *= i
    v_append[i,] *= i


kv_append_length = torch.tensor(num_tokens, dtype=torch.int32, device=device)
kv_append_indptr = torch.cat(
    [torch.zeros(1).int().to(0), torch.cumsum(kv_append_length, dim=0)]
).int()


batch_size = 4
num_tokens = [2, 6, 7, 16]
num_all_tokens = sum(num_tokens) # 100

required_page_indices = np.ceil([(num_token+1) / page_size for num_token in num_tokens])
kv_last_page_len = [(num_token) % page_size for num_token in num_tokens]
# kv_last_page_len = np.where(kv_last_page_len == 0, 4, kv_last_page_len)

num_pages_per_req = torch.tensor(required_page_indices, dtype=torch.int32, device=device)
kv_page_indptr = torch.cat(
    [torch.zeros(1).int().to(device), torch.cumsum(num_pages_per_req, dim=0)]
).int()

kv_page_indices = torch.arange(required_page_indices.sum(), dtype=torch.int32, device=device)
kv_last_page_len = torch.tensor(kv_last_page_len, dtype=torch.int32, device=device)

print(f"================ append_paged_kv_cache ================")
print(f"  kv_append_length: {kv_append_length.shape}")
print(f"  v_append.shape: {v_append.shape}")
print(f"  kv_append_indptr: {kv_append_indptr.shape}")
print(f"  paged_kv_cache: {paged_kv_cache.shape}")
print(f"  kv_page_indices: {kv_page_indices.shape}")
print(f"  kv_page_indptr: {kv_page_indptr.shape}")
print(f"  kv_last_page_len: {kv_last_page_len.shape}")
print(f"  ------")
print(f"  kv_append_length: {kv_append_length}")
print(f"  kv_append_indptr: {kv_append_indptr}")
print(f"  kv_page_indices: {kv_page_indices}")
print(f"  kv_page_indptr: {kv_page_indptr}")
print(f"  kv_last_page_len: {kv_last_page_len}")
print(f"========================================================")

flashinfer.append_paged_kv_cache(
    k_append,
    v_append,
    kv_append_indptr,
    paged_kv_cache,
    kv_page_indices,
    kv_page_indptr,
    kv_last_page_len
)

torch.cuda.synchronize()
print(f"================ result of append_paged_kv_cache ================")
print(f"  paged_kv_cache: {paged_kv_cache[0, 0, :, 0, 0]}")
print(f"  paged_kv_cache: {paged_kv_cache[1, 0, :, 0, 0]}")
print(f"  paged_kv_cache: {paged_kv_cache[2, 0, :, 0, 0]}")
print(f"  paged_kv_cache: {paged_kv_cache[3, 0, :, 0, 0]}")
print(f"  paged_kv_cache: {paged_kv_cache[4, 0, :, 0, 0]}")
print(f"  paged_kv_cache: {paged_kv_cache[5, 0, :, 0, 0]}")
print(f"  paged_kv_cache: {paged_kv_cache[6, 0, :, 0, 0]}")
print(f"  paged_kv_cache: {paged_kv_cache[7, 0, :, 0, 0]}")
print(f"  paged_kv_cache: {paged_kv_cache[8, 0, :, 0, 0]}")
print(f"========================================================")



qo_indptr = kv_append_indptr
q_at_layer = torch.randn(num_all_tokens, num_qo_heads, head_dim).half().to(device)



print(f"================ prefill_wrapper ================")
print(f"  q_at_layer: {q_at_layer.shape}")
print(f"  kv_append_indptr: {kv_append_indptr}")
print(f"  kv_page_indices: {kv_page_indices}")
print(f"  kv_page_indptr: {kv_page_indptr}")
print(f"  kv_last_page_len: {kv_last_page_len}")
print(f"========================================================")

# create auxiliary data structures for batch prefill attention
prefill_wrapper.plan(
    qo_indptr,
    kv_page_indptr,
    kv_page_indices,
    kv_last_page_len, # 마지막 page에서 토큰 찬 길이 context_len % page_size
    num_qo_heads, # fix
    num_kv_heads, # fix
    head_dim, # fix
    page_size, # fix
    causal=True,
)

# compute batch prefill attention, reuse auxiliary data structures
prefill_output = prefill_wrapper.run(q_at_layer, paged_kv_cache)


print(f"================ result of prefill_wrapper ================")
print(f"  prefill_output: {prefill_output.shape}")
print(f"========================================================")



print(f"================ decode_wrapper ================")
print(f"  q_at_layer: {q_at_layer.shape}")
print(f"  kv_append_indptr: {kv_append_indptr}")
print(f"  kv_page_indices: {kv_page_indices}")
print(f"  kv_page_indptr: {kv_page_indptr}")
print(f"  kv_last_page_len: {kv_last_page_len}")
print(f"========================================================")

# create auxiliary data structures for batch prefill attention
decode_wrapper.plan(
    kv_page_indptr,
    kv_page_indices,
    kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    pos_encoding_mode="NONE",
    data_type=torch.float16
)

decode_output = decode_wrapper.run(q_at_layer, paged_kv_cache)


print(f"================ prefill_wrapper ================")
print(f"  decode_output: {decode_output.shape}")
print(f"========================================================")