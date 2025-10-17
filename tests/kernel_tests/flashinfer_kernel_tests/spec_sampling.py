import torch

from flashinfer.sampling import chain_speculative_sampling

def run_chain_speculative_sampling(draft_probs, draft_token_ids, verify_probs):
    batch_size, num_draft_tokens = draft_token_ids.shape
    uniform_samples = torch.rand(batch_size, num_draft_tokens + 1, device=draft_token_ids.device)
    
        # print(f"\n------------- chain_speculative_sampling -------------")
        # print(f"batch_size: {batch_size}")
        # print(f"uniform_samples.shape: {uniform_samples.shape}")
        # print(f"draft_token_ids: {draft_token_ids}")
        # print(f"draft_probs.shape: {draft_probs.shape}")
        # print(f"draft_token_ids.shape: {draft_token_ids.shape}")
        # print(f"verify_probs.shape: {verify_probs.shape}")
        # print(f"------------------------------------------------------")

    accepted_token_ids, output_accepted_token_num, output_emitted_token_num =\
        chain_speculative_sampling(draft_probs, draft_token_ids, uniform_samples, verify_probs)
    

    return accepted_token_ids, output_accepted_token_num, output_emitted_token_num


token_ids = torch.tensor([[ 1,  0,  1, -1, -1]], device='cuda:0')
draft_probs = torch.tensor([[[0.5883, 0.4117, 0.0000, 0.0000],
                             [1.0000, 0.0000, 0.0000, 0.0000],
                             [0.3803, 0.5435, 0.0762, 0.0000],
                             [0.0000, 0.0000, 0.0000, 0.0000],
                             [0.0000, 0.0000, 0.0000, 0.0000]]], device='cuda:0')
verify_probs = torch.tensor([[[0.4555, 0.5445, 0.0000, 0.0000],
                             [1.0000, 0.0000, 0.0000, 0.0000],
                             [0.5783, 0.2831, 0.0000, 0.1386],
                             [0.2346, 0.6850, 0.0804, 0.0000],
                             [0.0000, 0.0000, 0.0000, 0.0000],
                             [0.0000, 0.0000, 0.0000, 0.0000]]], device='cuda:0')
verify_probs = torch.tensor([[[0.4555, 0.5445, 0.0000, 0.0000],
                             [1.0000, 0.0000, 0.0000, 0.0000],
                             [0.5783, 0.2831, 0.0000, 0.1386],
                             [0.0000, 0.0000, 0.0000, 0.0000],
                             [0.0000, 0.0000, 0.0000, 0.0000],
                             [0.0000, 0.0000, 0.0000, 0.0000]]], device='cuda:0')

accepted_token_ids, output_accepted_token_num, output_emitted_token_num = run_chain_speculative_sampling(draft_probs, token_ids, verify_probs)

print(accepted_token_ids)
print(output_accepted_token_num)
print(output_emitted_token_num)