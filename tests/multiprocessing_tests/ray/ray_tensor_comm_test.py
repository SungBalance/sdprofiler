import os, pickle
import ray
import nvtx
import torch


@ray.remote(num_gpus=0.5)
class WorkQueue:
    def __init__(self, name, data_dict):

        print(torch.cuda.is_available())
        self.queue = []
        self.name = name
        self.queue = [(key, val) for key, val in data_dict.items()]

    def get_work_item(self):
        if self.queue:
            return self.queue.pop(0)
        else:
            return None
        
    def get_name(self):
        return self.name


@ray.remote(num_gpus=0.5)
class WorkerWithoutPipelining:
    def __init__(self, work_queue):
        self.work_queue = work_queue
        self.name = ray.get(work_queue.get_name.remote())

    def process(self, work_item):
        (tag, data) = work_item
        print(tag)
        print(data)

    def run(self):

        with nvtx.annotate(f"{self.name}_run_without_pipelining"):
            while True:
                # Get work from the remote queue.

                with nvtx.annotate(f"{self.name}_run_without_pipelining_get"):
                    work_item = ray.get(self.work_queue.get_work_item.remote())

                if work_item is None:
                    break

                # Do work.
                self.process(work_item)


@ray.remote
class WorkerWithPipelining:
    def __init__(self, work_queue):
        self.work_queue = work_queue
        self.name = ray.get(work_queue.get_name.remote())

    def process(self, work_item):
        (tag, data) = work_item
        print(tag)
        print(data)

    def run(self):
        with nvtx.annotate(f"{self.name}_run_with_pipelining"):
            self.work_item_ref = self.work_queue.get_work_item.remote()

            while True:
                # Get work from the remote queue.
                with nvtx.annotate(f"{self.name}_run_with_pipelining_get"):
                    work_item = ray.get(self.work_item_ref)

                if work_item is None:
                    break

                self.work_item_ref = self.work_queue.get_work_item.remote()

                # Do work while we are fetching the next work item.
                self.process(work_item)


######


# get list of *.pkl files in the current directory
files = [f for f in os.listdir('.') if f.endswith('.pkl')]

data_dict = {}
for file in files:
    # load files and serialize to byte, then get the size of the byte
    with open(file, 'rb') as f:
        # data = torch.load(f, map_location={'cuda:1': 'cpu'})
        data = pickle.load(f)
        tag = ''.join(file.split('.')[0].split('_')[:2])
        data_dict[file] = data

for name in ['model_inputs', 'objs']:

    data_sample_dict = {key:val in data_dict.items() for key, val in data_dict.items() if name in key}
    
    ray.init()
    work_queue = WorkQueue.remote(name, data_sample_dict)
    worker_without_pipelining = WorkerWithoutPipelining.remote(work_queue) # actor

    ray.get(worker_without_pipelining.run.remote())


    ######

    work_queue = WorkQueue.remote(name)
    worker_with_pipelining = WorkerWithPipelining.remote(work_queue) # actor

    ray.get(worker_with_pipelining.run.remote())

    ray.shutdown()