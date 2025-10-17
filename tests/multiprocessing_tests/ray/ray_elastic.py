import time
import os
import atexit

import torch
from torch import distributed as dist

import ray
from ray.util.queue import Queue

@ray.remote(num_gpus=1)
class InferenceActor:
    def __init__(self, gpu_num):
        self.gpu_num = gpu_num
        self.device = f"cuda:0"
        torch.cuda.set_device(self.device)
    
    def check_device(self):
        print(f"ACTOR {self.gpu_num} :: check_device :: torch.cuda.device(0): {torch.cuda.device(0)}")
        print(f"ACTOR {self.gpu_num} :: check_device :: get_world_size: {torch.distributed.get_world_size()}")

    def get_gpu_num(self):
        return self.gpu_num
    
    def set_torch_distributed(self, world_size):
        start_time = time.time()
        if dist.is_initialized():
            self.reset_torch_distributed(world_size)
            pass
        else:
            self.create_torch_distributed(world_size)
        end_time = time.time()

        print(f"ACTOR {self.gpu_num} :: set_torch_distributed :: time: {end_time - start_time}, world_size: {world_size}")

    def create_torch_distributed(self, world_size):

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "135"
        os.environ["RANK"] = str(self.gpu_num)
        os.environ["WORLD_SIZE"] = str(world_size)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_num)

        print(f"ACTOR {self.gpu_num} :: set_torch_distributed :: gpu_num: {self.gpu_num}, world_size: {world_size}")
        dist.init_process_group(backend="nccl")
        print(f"ACTOR {self.gpu_num} :: init_process_group ends")

    def reset_torch_distributed(self, world_size):
        dist.destroy_process_group()
        print(f"ACTOR {self.gpu_num} :: reset_torch_distributed :: gpu_num: {self.gpu_num}, world_size: {world_size}")

        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group(backend="nccl")
        print(f"ACTOR {self.gpu_num} :: init_process_group ends")



    def infer(self, input_data):
        self.check_device()
        tensor = torch.tensor([input_data], device=self.device)
        tensor = tensor.repeat(1024, 1024)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor[0,0].item()

class RayElasticTaskManager:
    def __init__(self, num_gpus=4):
        ray.init(num_gpus=num_gpus)
        self.max_world_size = num_gpus
        self.gpu_pool = list(range(4))
        self.task_queue = Queue()
        self.workers = []
        print(f"RayElasticTaskManager initialized with gpu_pool: {self.gpu_pool}")

    def add_workers(self, num_workers):
        # create new workers
        new_workers = []
        for _ in range(min(num_workers, len(self.gpu_pool))):
            gpu_idx = self.gpu_pool.pop(0)
            new_worker = InferenceActor.remote(gpu_idx)
            self.workers.append(new_worker)
        
        # new_world_size = len(self.workers) + len(new_workers)
        # self.workers.extend(new_workers)
        new_world_size = len(self.workers)

        # initialize torch.distributed
        futures = []
        for worker in self.workers:
            futures.append(worker.set_torch_distributed.remote(new_world_size))
        ray.get(futures)

        print(f"Added {num_workers} workers, new world size: {new_world_size} gpu_pool: {self.gpu_pool}")

    def remove_workers(self, num_workers):
        for _ in range(num_workers):
            worker = self.workers.pop()
            self.gpu_pool.append(ray.get(worker.get_gpu_num.remote()))
            ray.kill(worker)

        futures = []
        new_world_size = len(self.workers)
        for worker in self.workers:
            futures.append(worker.set_torch_distributed.remote(new_world_size))
        ray.get(futures)

        print(f"Removed {num_workers} workers, gpu_pool: {self.gpu_pool}")

    def run(self):
        start_time = time.time()
        futures = []
        while not self.task_queue.empty():
            task = self.task_queue.get()
            # worker = self.workers.pop(0)
            for worker in self.workers:
                futures.append(worker.infer.remote(task))
            # self.workers.append(worker)
        result = ray.get(futures)
        end_time = time.time()
        print(f"num_workers: {len(self.workers)} time: {end_time - start_time}, result: {result}\n\n")

def shutdown_ray():
    print("Shutting down Ray...")
    ray.shutdown()

if __name__ == "__main__":

    NUM_TASKS = 1

    # Register the signal handler for termination signals
    # atexit.register(shutdown_ray)


    os.environ["RAY_DEDUP_LOGS"] = "0"

    manager = RayElasticTaskManager()


    print("##################################")

    manager.add_workers(2)
    for i in range(NUM_TASKS):
        manager.task_queue.put(100)
    manager.run()

    print("##################################")


    manager.add_workers(2)
    for i in range(NUM_TASKS):
        manager.task_queue.put(100)
    manager.run()

    print("##################################")

    manager.remove_workers(1)
    for i in range(NUM_TASKS):
        manager.task_queue.put(100)
    manager.run()

    print("##################################")
    ray.shutdown()