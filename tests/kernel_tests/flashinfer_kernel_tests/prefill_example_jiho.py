import torch
import flashinfer
import numpy as np

num_layers = 30
num_qo_heads = 64
num_kv_heads = 16
head_dim = 128

page_size = 16



def cal_kv_indtr(batch_token_distribution, max_num_pages):
    current_kv_indptr = 0
    kv_indptr = []
    kv_last_page_len = []
    for x in batch_token_distribution:
        kv_indptr.append(current_kv_indptr)
        if x % page_size == 0:
            current_kv_indptr = current_kv_indptr + (x // page_size)
            kv_last_page_len.append(page_size)
        else:    
            current_kv_indptr = current_kv_indptr + (x // page_size) + 1
            kv_last_page_len.append(x % page_size)
    
    #kv_indptr.append(max_num_pages)
    kv_indptr.append(current_kv_indptr)
    return kv_indptr, kv_last_page_len


def skewed_batch_token_distribution(batch_size, total_context_len):
    weights = np.abs(np.random.normal(loc=2, scale=1, size=batch_size))
    ratios = weights / weights.sum()
    raw_values = ratios * total_context_len
    values = np.floor(raw_values).astype(int)
    
    remainder = total_context_len - values.sum()
    fractional = raw_values - values
    for i in np.argsort(-fractional)[:remainder]:
        values[i] += 1

    return values.tolist()

def hard_skewed_batch_token_distribution(batch_size, total_context_len):
    z = np.abs(np.random.normal(loc=0, scale=1, size=batch_size))
    weights = np.exp(z)
    ratios = weights / weights.sum()
    raw_values = ratios * total_context_len
    values = np.floor(raw_values).astype(int)
    
    remainder = total_context_len - values.sum()
    fractional = raw_values - values
    for i in np.argsort(-fractional)[:remainder]:
        values[i] += 1

    return values.tolist()

def normal_batch_token_distribution(batch_size, total_context_len):
    weights = np.abs(np.random.normal(loc=0, scale=1, size=batch_size))
    ratios = weights / weights.sum()
    raw_values = ratios * total_context_len
    values = np.floor(raw_values).astype(int)
    
    remainder = total_context_len - values.sum()
    fractional = raw_values - values
    for i in np.argsort(-fractional)[:remainder]:
        values[i] += 1

    return values.tolist()
    
def run(
        t,
        batch_size,
        nnz_qo,
        context_len_mean,
        total_context_len
    ):
    # allocate 128MB workspace buffer
    max_num_pages = (total_context_len + 6 * batch_size) // 7
    workspace_buffer = torch.empty(max_num_pages * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    
    # 각 배치마다 가지는 prefill + kv 토큰 개수
    if t == "same":
        batch_token_distribution = [context_len_mean + prefill_generate_token_size for i in range(batch_size)]
    elif t == "normal":
        batch_token_distribution = normal_batch_token_distribution(batch_size, total_context_len)
    elif t == "skewed":
        batch_token_distribution = skewed_batch_token_distribution(batch_size, total_context_len)
    elif t == "hard_skewed":
        batch_token_distribution = hard_skewed_batch_token_distribution(batch_size, total_context_len)
        
    kv_indptr, kv_last_page_len = cal_kv_indtr(batch_token_distribution, max_num_pages)
    
    
    qo_indptr = torch.tensor(
        [x * 6 for x in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    
    
    paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
    paged_kv_indptr = torch.tensor(
        kv_indptr, dtype=torch.int32, device="cuda:0"
    )
    # 1 <= paged_kv_last_page_len <= page_size
    paged_kv_last_page_len = torch.tensor(
        kv_last_page_len, dtype=torch.int32, device="cuda:0"
    )
    
    q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
    kv_cache_at_layer = torch.randn(
        num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
    )
    
    # create auxiliary data structures for batch prefill attention
    prefill_wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len, # 마지막 page에서 토큰 찬 길이 context_len % page_size
        num_qo_heads, # fix
        num_kv_heads, # fix
        head_dim, # fix
        page_size, # fix
        causal=True,
    )
    outputs = []
    
    for i in range(3):
        q = q_at_layer[i]
        kv_cache = kv_cache_at_layer[i]
        # compute batch prefill attention, reuse auxiliary data structures
        o = prefill_wrapper.run(q, kv_cache)

    for i in range(num_layers):
        q = q_at_layer[i]
        kv_cache = kv_cache_at_layer[i]
        # compute batch prefill attention, reuse auxiliary data structures
        starter.record()
        o = prefill_wrapper.run(q, kv_cache)
        ender.record()
        torch.cuda.synchronize()
        infer_time = starter.elapsed_time(ender)
        times.append(infer_time)
        outputs.append(o)
        

    mean_times = np.array(times).mean()
    dic = {
        "type": t,
        "batch_size": batch_size,
        "context_len_mean": context_len_mean,
        "mean_time": mean_times,
        "prefill_token": outputs[0].shape[0] // batch_size
        
    }
    print(f"=====================")
    print(f"type : {t}")
    print(f"batch_size : {batch_size}")
    print(f"context_len_mean : {context_len_mean}")
    print("Mean times :", mean_times)
    print("outputs[0].shape :", outputs[0].shape)
    
    del qo_indptr, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len, q_at_layer, kv_cache_at_layer, workspace_buffer
    torch.cuda.empty_cache()
    
    return dic


if __name__ == '__main__':
    # 16, 32, 64
    batch_size = [16, 32, 64]
    prefill_generate_token_size = 6
    nnz_qo = [prefill_generate_token_size * x for x in batch_size]
    
    results = []
    
    for b, n in zip(batch_size, nnz_qo):
        #64, 128, 256, 512
        context_len_mean = [64, 128, 256, 512]
        total_context_len = [x * b for x in context_len_mean]
        for c, total_c in zip(context_len_mean, total_context_len):
            results.append(run("same", b, n, c, total_c))
            results.append(run("normal", b, n, c, total_c))
            results.append(run("skewed", b, n, c, total_c))
            results.append(run("hard_skewed", b, n, c, total_c))
            
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("output_new.csv")
    
#qo_indtr 은 6으로 조절
#qo_indptr = torch.tensor(
#    [x * 6 for x in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
#)
#paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")
#paged_kv_indptr = torch.tensor(
#    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
#)
## 1 <= paged_kv_last_page_len <= page_size
#paged_kv_last_page_len = torch.tensor(
#    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
#)
#q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")
#kv_cache_at_layer = torch.randn(
#    num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
#)
#
## create auxiliary data structures for batch prefill attention
#prefill_wrapper.plan(
#    qo_indptr,
#    paged_kv_indptr,
#    paged_kv_indices,
#    paged_kv_last_page_len, # 마지막 page에서 토큰 찬 길이 context_len % page_size
#    num_qo_heads, # fix
#    num_kv_heads, # fix
#    head_dim, # fix
#    page_size, # fix
#    causal=True,
#)
#outputs = []
#
#for i in range(num_layers):
#    q = q_at_layer[i]
#    kv_cache = kv_cache_at_layer[i]
#    # compute batch prefill attention, reuse auxiliary data structures
#    o = prefill_wrapper.run(q, kv_cache)
#    outputs.append(o)
#
#outputs[0].shape
## torch.Size([100, 64, 128])
#
## below is another example of creating custom mask for batch prefill attention
#mask_arr = []
#qo_len = (qo_indptr[1:] - qo_indptr[:-1]).cpu().tolist()
#kv_len = (page_size * (paged_kv_indptr[1:] - paged_kv_indptr[:-1] - 1) + paged_kv_last_page_len).cpu().tolist()
#for i in range(batch_size):
#    mask_i = torch.tril(
#        torch.full((qo_len[i], kv_len[i]), True, device="cuda:0"),
#        diagonal=(kv_len[i] - qo_len[i]),
#    )
#    mask_arr.append(mask_i.flatten())
#
#mask = torch.cat(mask_arr, dim=0)
#prefill_wrapper.plan(
#    qo_indptr,
#    paged_kv_indptr,
#    paged_kv_indices,
#    paged_kv_last_page_len,
#    num_qo_heads,
#    num_kv_heads,
#    head_dim,
#    page_size,
#    custom_mask=mask,
#)
#for i in range(num_layers):
#    q = q_at_layer[i]
#    kv_cache = kv_cache_at_layer[i]
#    # compute batch prefill attention, reuse auxiliary data structures
#    o_custom = prefill_wrapper.run(q, kv_cache)
#    assert torch.allclose(o_custom, outputs[i], rtol=1e-3, atol=1e-3)
