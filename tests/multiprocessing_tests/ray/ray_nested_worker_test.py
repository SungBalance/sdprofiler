import os, pickle
import ray
import nvtx
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor


@ray.remote(num_gpus=1)
class CommWorker:
    def __init__(self):
        self.counter = 0
        self.device = 'cuda:0'
        torch.cuda.set_device(self.device)

        self.queue = []
        for _ in range(10):
            self.queue.append(self.create_tensor())

    def create_tensor(self):
        return torch.empty((10240, 10240), device=self.device)
    
    def export_tensor(self, tensor):
        storage = tensor.storage()
        
        (storage_device, storage_handle, storage_size_bytes, storage_offset_bytes,
        ref_counter_handle, ref_counter_offset, event_handle, event_sync_required) = storage._share_cuda_()

        return {
            "dtype": tensor.dtype,
            "tensor_size": tensor.size(),
            "tensor_stride": tensor.stride(),
            "tensor_offset": tensor.storage_offset(), # !Not sure about this one.
            "storage_cls": type(storage),
            "storage_device": storage_device,
            "storage_handle": storage_handle,
            "storage_size_bytes": storage_size_bytes,
            "storage_offset_bytes": storage_offset_bytes,
            "requires_grad": False,
            "ref_counter_handle": ref_counter_handle,
            "ref_counter_offset": ref_counter_offset,
            "event_handle": event_handle,
            "event_sync_required": event_sync_required,
        }
    
    def execute_comm(self):
        tensor = self.queue.pop(0)
        ipc_data = self.export_tensor(tensor)
        print([item.shape for item in self.queue])
        print(f"EXPORTED: {torch.cuda.list_gpu_processes()}")
        return ipc_data
    
    def get_device(self):
        return self.device


@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, comm_worker):
        self.comm_worker = comm_worker
        self.device = ray.get(comm_worker.get_device.remote())
        torch.cuda.set_device(self.device)
        self.queue = []

    def process(self, tensor):
        print([item.shape for item in self.queue])
        print(f"IMPORTED: {torch.cuda.list_gpu_processes()}")
        
        self.queue.append(tensor)

    def import_tensor(self, ipc_data):
        return rebuild_cuda_tensor(torch.Tensor, **ipc_data)

    def run(self):
        with nvtx.annotate(f"run_get"):
            ipc_data = ray.get(self.comm_worker.execute_comm.remote())
        if ipc_data is None:
            return None
        # Do work.
        tensor = self.import_tensor(ipc_data)
        self.process(tensor)
        return 0


if __name__ == '__main__':
    print(f"BEFORE RAY INIT, {torch.cuda.list_gpu_processes()}")
    comm_actor = CommWorker.remote()
    ray.get(comm_actor.__ray_ready__.remote())

    worker = Worker.remote(comm_actor) # actor
    ray.get(worker.__ray_ready__.remote())

    print(f"INITIALIZED, {torch.cuda.list_gpu_processes()}")

    for i in range(10):
        result = ray.get(worker.run.remote())
        input(f'result: {result}')
    print("GET DONE")
    input()
    ray.shutdown()
    print("SHUTDOWN")