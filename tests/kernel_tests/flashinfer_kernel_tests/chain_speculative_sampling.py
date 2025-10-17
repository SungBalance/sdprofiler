import torch
import flashinfer

torch.manual_seed(42)
batch_size = 1
num_speculate_tokens = 4
vocab_size = 4

draft_token_ids = torch.tensor([[3, 3, 3, 3]], dtype=torch.int32).to(0)

draft_probs = torch.tensor(
    [[[0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0]]]).to(0)

target_probs = torch.tensor(
    [[[1.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0], 
      [1.0, 0.0, 0.0, 0.0], 
      [1.0, 0.0, 0.0, 0.0], 
      [1.0, 0.0, 0.0, 0.0]]]).to(0)


# uniform samples for rejection sampling
uniform_samples = torch.rand(batch_size, num_speculate_tokens + 1).to(0)

output_token_ids, output_accepted_token_num, output_emitted_token_num =\
    flashinfer.sampling.chain_speculative_sampling(
        draft_probs, draft_token_ids, uniform_samples, target_probs)

print(f"0 accepted")
print(f"output_token_ids: {output_token_ids}")
print(f"output_accepted_token_num: {output_accepted_token_num}")
print(f"output_emitted_token_num: {output_emitted_token_num}")

print("--------------------------------")

draft_probs = torch.tensor(
    [[[0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0]]]).to(0)

target_probs = torch.tensor(
    [[[0.0, 0.0, 0.0, 1.0],
      [1.0, 0.0, 0.0, 0.0], 
      [1.0, 0.0, 0.0, 0.0], 
      [1.0, 0.0, 0.0, 0.0], 
      [1.0, 0.0, 0.0, 0.0]]]).to(0)


# uniform samples for rejection sampling
uniform_samples = torch.rand(batch_size, num_speculate_tokens + 1).to(0)

output_token_ids, output_accepted_token_num, output_emitted_token_num =\
    flashinfer.sampling.chain_speculative_sampling(
        draft_probs, draft_token_ids, uniform_samples, target_probs)

print(f"1 accepted")
print(f"output_token_ids: {output_token_ids}")
print(f"output_accepted_token_num: {output_accepted_token_num}")
print(f"output_emitted_token_num: {output_emitted_token_num}")

print("--------------------------------")

draft_probs = torch.tensor(
    [[[0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0]]]).to(0)

target_probs = torch.tensor(
    [[[0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0], 
      [0.0, 0.0, 0.0, 1.0], 
      [1.0, 0.0, 0.0, 0.0], 
      [1.0, 0.0, 0.0, 0.0]]]).to(0)


# uniform samples for rejection sampling
uniform_samples = torch.rand(batch_size, num_speculate_tokens + 1).to(0)

output_token_ids, output_accepted_token_num, output_emitted_token_num =\
    flashinfer.sampling.chain_speculative_sampling(
        draft_probs, draft_token_ids, uniform_samples, target_probs)

print(f"3 accepted")
print(f"output_token_ids: {output_token_ids}")
print(f"output_accepted_token_num: {output_accepted_token_num}")
print(f"output_emitted_token_num: {output_emitted_token_num}")

print("--------------------------------")


draft_probs = torch.tensor(
    [[[0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0]]]).to(0)

target_probs = torch.tensor(
    [[[0.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0], 
      [0.0, 0.0, 0.0, 1.0], 
      [0.0, 0.0, 0.0, 1.0], 
      [1.0, 0.0, 0.0, 0.0]]]).to(0)

output_token_ids, output_accepted_token_num, output_emitted_token_num =\
    flashinfer.sampling.chain_speculative_sampling(
        draft_probs, draft_token_ids, uniform_samples, target_probs)

print(f"4 accepted")
print(f"output_token_ids: {output_token_ids}")
print(f"output_accepted_token_num: {output_accepted_token_num}")
print(f"output_emitted_token_num: {output_emitted_token_num}")
