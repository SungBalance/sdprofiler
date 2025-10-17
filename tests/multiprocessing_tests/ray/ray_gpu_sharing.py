import os
import time

import ray
import torch

runtime_env={
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": "0,1"
        }
    }


class GPUActor:
    def __init__(self, gpu_local_rank: int):
        self.gpu_local_rank = gpu_local_rank
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_local_rank}"
        
        assigned_gpus = ray.get_gpu_ids()
        
        ctx = ray.get_runtime_context()
        self.actor_id = ctx.get_actor_id()
        
        self.device = torch.device(f"cuda:0")
        self.stream = torch.cuda.Stream(self.device)

        print(f"[Actor __init__] actor_id: {self.actor_id}," \
              f"local_rank: {gpu_local_rank}," \
              f"torch device: {torch.cuda.current_device()}," \
              f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}," \
              f"ray_assigned_gpus: {assigned_gpus}")

    def run_matmul(self, input_tensor :torch.Tensor=None):
        print(f"Before run_matmul")
        print(torch.cuda.memory_summary())
        size = input_tensor.shape[0]
        weight_tensor = torch.randn(size, size, device=self.device)
        
        start = time.time()
        with torch.cuda.stream(self.stream):
            output_tensor = torch.matmul(input_tensor, weight_tensor)
        torch.cuda.synchronize(self.device)
        end = time.time()
        print(f"After run_matmul")
        print(torch.cuda.memory_summary())
        return output_tensor

    def do_work(self, size=2000):
        start = time.time()

        with torch.cuda.stream(self.stream):
            x = torch.randn(size, size, device=self.device)
            y = torch.randn(size, size, device=self.device)
            z = x @ y
        
        torch.cuda.synchronize(self.device)
        end = time.time()
        output = f"actor_id: {self.actor_id}, " \
                f"gpu_local_rank: {self.gpu_local_rank}, " \
                f"torch.cuda.device: {self.device}, " \
                f"time: {end - start:.4f} seconds"

        return output

if __name__ == "__main__":

    os.environ["RAY_DEDUP_LOGS"] = "0"

    num_available_gpus = torch.cuda.device_count()
    ray_custom_resources={
        f"GPU_{i}": 1 for i in range(num_available_gpus)
    }

    ray.init(
        resources=ray_custom_resources
    )

    runtime_env["nsight"] = {
                "t": "cuda,osrt,nvtx,cudnn,cublas",
                "cuda-graph-trace": "node",
                "o": "worker_process_%p",
                "w": "true",
                "f": "true",
            }

    actors = [
        ray.remote(num_cpus=0, resources={f"GPU_{0}": 0.5}, runtime_env=runtime_env)(GPUActor).remote(gpu_local_rank=0),
        ray.remote(num_cpus=0, resources={f"GPU_{1}": 0.5}, runtime_env=runtime_env)(GPUActor).remote(gpu_local_rank=1),
        # ray.remote(num_cpus=0, resources={f"GPU_{0}": 0.5})(GPUActor).remote(gpu_local_rank=0),
    ]


    print(f"run_do_work_remote")

    futures = [actor.do_work.remote(3000) for actor in actors]
    results = ray.get(futures)

    
    input_tensor = torch.randn(1024*16, 1024*16).cuda(0)
    print(f"Before run_matmul")
    print(torch.cuda.memory_summary())
    torch.cuda.synchronize()
    futures = [actor.run_matmul.remote(input_tensor) for actor in actors]
    results = ray.get(futures)
    print(f"After run_matmul")
    print(torch.cuda.memory_summary())
    
    for result in results:
        print(result.shape)

    """ outputs
    actor_id: 99d87cb1f179989836897a6801000000, gpu_local_rank: 0, torch.cuda.device: cuda:0, time: 0.0683 seconds
    actor_id: 204a83ada856fef0efbaef1901000000, gpu_local_rank: 1, torch.cuda.device: cuda:0, time: 0.0720 seconds
    actor_id: a127eb56e2bf38282866b6bc01000000, gpu_local_rank: 0, torch.cuda.device: cuda:0, time: 0.0707 seconds
    """

    ray.shutdown()


