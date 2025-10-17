import os

import torch
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

import nvtx

os.environ["RAY_COLOR_PREFIX"] = "1"
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29511'

def set_cuda_visible_devices(device_ids) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))

HIDDEN_SIZE = 1024

class Worker:
    def __init__(self, worker_id, num_workers):
        if worker_id == 0:
            self.is_driver_worker = True
        else:
            self.is_driver_worker = False
        
        self.worker_id = worker_id
        self.num_workers = num_workers
        

    def set_gpu(self):
        torch.cuda.set_device(self.worker_id)
        self.device = f'cuda'
        print(f":: Worker {self.worker_id} :: device: {self.device}, GPU: {torch.cuda.current_device()} / {torch.cuda.is_available()}")
    
    def init_model(self):
        self.weight = torch.rand(HIDDEN_SIZE, 10240, device=self.device)
        self.weight_2 = torch.rand(10240, HIDDEN_SIZE, device=self.device)

    def set_distributed(self):
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.num_workers,
            rank=self.worker_id,
        )

    def create_input_data(self):        
        something_python = [3] * HIDDEN_SIZE
        something_cuda = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, device=self.device)
        return something_python, something_cuda


    def send_tensor(self, tensor, dst):
        torch.distributed.send(tensor, dst)


    def receive_tensor(self, src):
        tensor = torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, device=self.device)
        torch.distributed.recv(tensor, src)
        return tensor

    @nvtx.annotate("Worker.forward")
    def forward(self, something_python, input_tensor):
        hidden = torch.matmul(self.weight, self.weight_2)
        hidden = hidden[0,]
        input_tensor[self.worker_id,] = hidden
        return input_tensor


    def work(self):
        if self.worker_id == 0:
            input_python, input_tensor = self.create_input_data()
        else:
            receive_src = self.worker_id - 1
            input_tensor = self.receive_tensor(receive_src)
            
        output_tensor = self.forward(None, input_tensor)
        
        if self.worker_id != self.num_workers - 1:
            send_dst = self.worker_id + 1
            self.send_tensor(input_tensor, send_dst)
        return output_tensor


    def exit_actor(self):
        print(f":: Worker {self.worker_id} :: Exiting actor")
        ray.actor.exit_actor()



class RayWorkerController:
    def __init__(self, num_workers=2):
        self.num_workers = num_workers

        # INIT RAY ========================================
        ray.init()
        placement_group_specs = ([{"GPU": 1}] * self.num_workers)
        current_placement_group = ray.util.placement_group(
            placement_group_specs)
        ray.get(current_placement_group.ready(), timeout=1800)

        print(f":: RayWorkerController :: current_placement_group: {current_placement_group}")
        print(f":: RayWorkerController :: num_gpus_in_cluster: {ray.cluster_resources().get('GPU', 0)}")
        
        # INIT WORKERS ====================================

        # self.workers  = [ray.remote(num_cpus=0, num_gpus=1)(Worker).remote(i, self.num_workers) for i in range(1, self.num_workers)]

        self.workers = []
        for bundle_id, bundle in enumerate(current_placement_group.bundle_specs):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=current_placement_group,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_id,
                )
            
            self.workers.append(
                ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=scheduling_strategy,
                )(Worker).remote(bundle_id, self.num_workers)
            )

        del self.workers[0]
        self.driver_worker = Worker(0, self.num_workers)

        self.set_gpu()
        self.set_distributed()
        self.set_model()
        print(":: RayWorkerController :: Workers created")
        print(":: RayWorkerController :: torch.distributed is set")

    def set_gpu(self):
        ray_worker_outputs = [
            worker.set_gpu.remote()
            for worker in self.workers
        ]
        self.driver_worker.set_gpu()
        
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)
    
    def set_distributed(self):
        ray_worker_outputs = [
            worker.set_distributed.remote()
            for worker in self.workers
        ]
        self.driver_worker.set_distributed()
        
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

    def init_model(self):
        ray_worker_outputs = [
            worker.init_model.remote()
            for worker in self.workers
        ]
        self.driver_worker.init_model()
        
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

    @nvtx.annotate("RayWorkerController.run")
    def run(self):
        for i in range(self.num_workers):

            ray_worker_outputs = [
                worker.work.remote()
                for worker in self.workers
            ]
            driver_worker_output = self.driver_worker.work()
            
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return [driver_worker_output] + ray_worker_outputs


    def kill_actor(self):
        for worker in self.workers:
            worker.exit_actor.remote()


if __name__ == "__main__":
    controller = RayWorkerController(num_workers=2)
    
    outputs = controller.run()

    for idx, output in enumerate(outputs):
        print(f":: Main :: output {idx} :: \n{output}")
    controller.kill_actor()
    ray.shutdown()