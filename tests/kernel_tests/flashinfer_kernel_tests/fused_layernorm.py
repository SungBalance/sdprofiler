import flashinfer
import torch
import torch.nn as nn

dtype = torch.float16
hidden_size = 1024

hidden_states = torch.randn(1, hidden_size, dtype=dtype, device="cuda")
residual = torch.randn(1, hidden_size, dtype=dtype, device="cuda")
variance_epsilon = 1e-8


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype="float16"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
rmsnorm = Qwen2RMSNorm(hidden_size, eps=variance_epsilon, dtype=dtype).to("cuda")

hf_output = rmsnorm(hidden_states + residual)
flashinfer.norm.fused_add_rmsnorm(hidden_states, residual, rmsnorm.weight, variance_epsilon)

print(f"hf_output: {hf_output.dtype}")
print(f"flashinfer_output: {hidden_states.dtype}")
print(torch.allclose(hf_output, hidden_states))
