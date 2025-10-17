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
    def __init__(self, GPUS: int, gpu_local_rank: int):
        self.gpu_local_rank = gpu_local_rank
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPUS}"
        
        assigned_gpus = ray.get_gpu_ids()
        
        ctx = ray.get_runtime_context()
        self.actor_id = ctx.get_actor_id()
        
        self.device = torch.device(f"cuda:{self.gpu_local_rank}")
        self.stream = torch.cuda.Stream(self.device)

        print(f"[Actor __init__] actor_id: {self.actor_id}," \
              f"local_rank: {gpu_local_rank}," \
              f"torch device: {torch.cuda.current_device()}," \
              f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}," \
              f"ray_assigned_gpus: {assigned_gpus}")


    def do_work(self, size=2000):
        start = time.time()

        with torch.cuda.stream(self.stream):
            x = torch.randn(size, size, device=self.device)
            y = torch.randn(size, size, device=self.device)
            z = x @ y
        
        torch.cuda.synchronize(self.device)
        
        end = time.time()
        return f"actor_id: {self.actor_id}, " \
            f"gpu_local_rank: {self.gpu_local_rank}, " \
            f"torch.cuda.device: {self.device}, " \
            f"time: {end - start:.4f} seconds"

if __name__ == "__main__":

    os.environ["RAY_DEDUP_LOGS"] = "0"

    ray.init()

    num_available_gpus = torch.cuda.device_count()
    print(f"num_available_gpus: {num_available_gpus}")
    GPUS = ",".join(str(i) for i in range(num_available_gpus))
    
    actors = [
        ray.remote(num_cpus=0)(GPUActor).remote(GPUS=GPUS, gpu_local_rank=0),
        ray.remote(num_cpus=0)(GPUActor).remote(GPUS=GPUS, gpu_local_rank=1),
        ray.remote(num_cpus=0)(GPUActor).remote(GPUS=GPUS, gpu_local_rank=0),
    ]

    futures = [actor.do_work.remote(3000) for actor in actors]
    
    results = ray.get(futures)
    
    for result in results:
        print(result)

    """ outputs
    actor_id: 99d87cb1f179989836897a6801000000, gpu_local_rank: 0, torch.cuda.device: cuda:0, time: 0.0683 seconds
    actor_id: 204a83ada856fef0efbaef1901000000, gpu_local_rank: 1, torch.cuda.device: cuda:0, time: 0.0720 seconds
    actor_id: a127eb56e2bf38282866b6bc01000000, gpu_local_rank: 0, torch.cuda.device: cuda:0, time: 0.0707 seconds
    """

    ray.shutdown()


