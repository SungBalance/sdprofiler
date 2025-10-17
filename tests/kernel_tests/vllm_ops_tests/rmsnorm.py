import torch
from vllm._custom_ops import rms_norm, fused_add_rms_norm


def test_rmsnorm():
    x = torch.randn(10, 10)
    weight = torch.randn(10)
    epsilon = 1e-5
    out = torch.empty_like(x)
    rms_norm(out, x, weight, epsilon)
    
    # Compare with PyTorch's RMS normalization
    torch_out = torch.nn.functional.layer_norm(x, x.shape[1:], weight=weight, eps=epsilon)
    assert torch.allclose(out, torch_out), "RMSNorm output does not match PyTorch's layer_norm output"
    

def test_fused_add_rmsnorm():
    x = torch.randn(10, 10)
    residual = torch.randn(10, 10)
    weight = torch.randn(10)
    epsilon = 1e-5
    out = torch.empty_like(x)
    fused_add_rms_norm(out, x, residual, weight, epsilon)
    
    # Compare with standard addition followed by RMS normalization
    torch_out = torch.nn.functional.layer_norm(x + residual, x.shape[1:], weight=weight, eps=epsilon)
    assert torch.allclose(out, torch_out), "FusedAddRMSNorm output does not match standard RMSNorm output"


if __name__ == "__main__":
    test_rmsnorm()
    test_fused_add_rmsnorm()
