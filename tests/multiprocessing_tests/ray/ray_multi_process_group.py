
import os
import ray
import torch
import torch.distributed as dist
import numpy as np

# rendezvous 설정
MASTER_ADDR = "127.0.0.1"
NCCL_PORT  = "29500"
GLOO_PORT  = "29501"

def init_process_group(backend: str, world_size: int, rank: int, port: str):
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = port
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


class GPUCommWorker:
    def __init__(self, GPUS: str, global_rank: int, local_rank: int):

        os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPUS}"
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

        print(f"GPUCommWorker.__init__: global_rank: {global_rank}, local_rank: {local_rank}")

        torch.cuda.set_device(local_rank)
        init_process_group("nccl", world_size=2, rank=local_rank, port=NCCL_PORT)

        self.global_rank = global_rank
        self.local_rank  = local_rank

    def run(self):
        x = torch.ones(4, device="cuda")
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        print(f"[GPUCommWorker gr{self.global_rank} lr{self.local_rank}] all_reduce →", x)

        return x.cpu().numpy()


class CPUCommWorker:
    def __init__(self, GPUS: str, global_rank: int, local_rank: int):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPUS}"

        print(f"CPUCommWorker.__init__: global_rank: {global_rank}, local_rank: {local_rank}")

        torch.cuda.set_device(local_rank)
        init_process_group("gloo", world_size=2, rank=local_rank, port=GLOO_PORT)

        self.global_rank = global_rank
        self.local_rank  = local_rank

    def run(self):
        y = torch.tensor([self.global_rank], device="cuda") * 3
        y_cpu = y.cpu().numpy()

        tensor = torch.from_numpy(y_cpu)
        dist.broadcast(tensor, src=0)
        result = tensor.numpy()
        print(f"[CPUCommWorker gr{self.global_rank} lr{self.local_rank}] broadcast →", result)

        return result

if __name__ == "__main__":

    num_available_gpus = torch.cuda.device_count()
    ray_custom_resources={
        f"GPU_{i}": 1 for i in range(num_available_gpus)
    }

    ray.init(
        resources=ray_custom_resources
    )


    num_available_gpus = torch.cuda.device_count()
    print(f"num_available_gpus: {num_available_gpus}")
    GPUS = ",".join(str(i) for i in range(num_available_gpus))

    gpu_workers = [
        ray.remote(num_cpus=0)(GPUCommWorker).remote(GPUS=GPUS, global_rank=0, local_rank=0),
        ray.remote(num_cpus=0)(GPUCommWorker).remote(GPUS=GPUS, global_rank=2, local_rank=1),
    ]

    cpu_workers = [
        ray.remote(num_cpus=0)(CPUCommWorker).remote(GPUS=GPUS, global_rank=1, local_rank=0),
        ray.remote(num_cpus=0)(CPUCommWorker).remote(GPUS=GPUS, global_rank=3, local_rank=1),
    ]

    gpu_results = ray.get([w.run.remote() for w in gpu_workers])
    cpu_results = ray.get([w.run.remote() for w in cpu_workers])

    print("\n\n--------------------------------")
    print("GPU group:", gpu_results)
    print("CPU group:", cpu_results)
    print("--------------------------------\n\n")

    ray.shutdown()