import numpy as np
def generate_array_constructive_np(num_all_tokens, num_draft_steps, batch_size):
    num_draft_tokens_to_execute_per_batch = np.zeros(batch_size, dtype=np.int32)

    num_full_queries = num_all_tokens // num_draft_steps
    if num_full_queries:
        num_draft_tokens_to_execute_per_batch[:num_full_queries] = num_draft_steps
        num_all_tokens -= num_full_queries * num_draft_steps

    remaining_queries = batch_size - num_full_queries
    if remaining_queries:
        arr = np.random.randint(0, remaining_queries, size=num_all_tokens)
        query_indices, counts = np.unique(arr, return_counts=True)
        query_indices += num_full_queries
        num_draft_tokens_to_execute_per_batch[query_indices] = counts

    # random
    # arr = np.random.randint(0, batch_size-1, size=num_all_tokens)
    # unique_values, counts = np.unique(arr, return_counts=True)
    # num_draft_tokens_to_execute_per_batch[unique_values] = counts

    return num_draft_tokens_to_execute_per_batch
if __name__ == "__main__":
    import pdb; pdb.set_trace()
