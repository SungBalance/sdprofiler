
import torch

def test_async_HtoD_copy(device: torch.device, copy_stream: torch.cuda.Stream, hidden_size: int = 1024):
    cpu_tensor = torch.randn(hidden_size, hidden_size, pin_memory=True)
    gpu_tensor = torch.zeros(hidden_size, hidden_size, device=device)
    matmul_tensor = torch.randn(hidden_size, hidden_size, device=device)
    
    # Async copy Device to Host
    with torch.cuda.stream(copy_stream):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)

    # Computation on Default Stream (only issue)
    with torch.cuda.stream(torch.cuda.current_stream()):
        matmul_tensor = torch.matmul(matmul_tensor, matmul_tensor)
    
    torch.cuda.synchronize() # Wait for computation


def test_async_DtoH_copy(device: torch.device, copy_stream: torch.cuda.Stream, hidden_size: int = 1024):
    cpu_tensor = torch.zeros(hidden_size, hidden_size, pin_memory=True)
    gpu_tensor = torch.randn(hidden_size, hidden_size, device=device)
    
    # Async copy Device to Host
    with torch.cuda.stream(copy_stream):
        cpu_tensor.copy_(gpu_tensor, non_blocking=True)
    
    # Computation on Default Stream (only issue)
    with torch.cuda.stream(torch.cuda.current_stream()):
        result = torch.matmul(gpu_tensor, gpu_tensor)
    
    torch.cuda.synchronize() # Wait for computation


def test_async_DtoH_copy_and_cpu_op(device: torch.device, copy_stream: torch.cuda.Stream, hidden_size: int = 1024):
    cpu_tensor = torch.zeros(hidden_size, hidden_size, pin_memory=True)
    gpu_tensor = torch.randn(hidden_size, hidden_size, device=device)
    
    # Async copy Device to Host
    with torch.cuda.stream(copy_stream):
        cpu_tensor.copy_(gpu_tensor, non_blocking=True)
    
    # Computation on Default Stream (only issue)
    with torch.cuda.stream(torch.cuda.current_stream()):
        result = torch.matmul(gpu_tensor, gpu_tensor)


    # Wait for copy and do CPU operation
    copy_stream.synchronize()
    with torch.cuda.nvtx.range("cpu_op"):
        size = cpu_tensor.shape[0] // 8 # reduce CPU operation overhead
        cpu_tensor = cpu_tensor[:size, :size]
        cpu_tensor = torch.matmul(cpu_tensor, cpu_tensor)
    
    torch.cuda.synchronize() # Wait for computation


if __name__ == "__main__":

    # Init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_size = 8192
    copy_stream = torch.cuda.Stream()

    # Warmup
    warmup_tensor = torch.randn(hidden_size, hidden_size, device=device)
    with torch.cuda.nvtx.range("warmup"):
        tensor_list = []
        for _ in range(3):
            tensor_list.append(torch.matmul(warmup_tensor, warmup_tensor))
    del warmup_tensor, _
    torch.cuda.synchronize()
    
    # Test
    with torch.cuda.nvtx.range("HtoD"):
        test_async_HtoD_copy(device=device, copy_stream=copy_stream, hidden_size=hidden_size)
    with torch.cuda.nvtx.range("DtoH"):
        test_async_DtoH_copy(device=device, copy_stream=copy_stream, hidden_size=hidden_size)
    with torch.cuda.nvtx.range("DtoH_and_cpu_op"):
        test_async_DtoH_copy_and_cpu_op(device=device, copy_stream=copy_stream, hidden_size=hidden_size)
