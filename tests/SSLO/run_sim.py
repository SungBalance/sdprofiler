
from typing import List, Union

import pandas as pd

class Request:
    def __init__(self, id: int, prompt_ids: List[int], response_ids: List[int]):
        self.id = id
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.latency = 0


class SSLOScheduler:
    def __init__(self, args):
        self.args = args
        self.waiting_requests = []
        self.scheduled_requests = []


    def add_request(self, requests: Union[Request, List[Request]]):
        if isinstance(requests, Request):
            requests = [requests]
        self.waiting_requests.extend(requests)


    def remove_request(self, request: Request):
        self.block_manager.deallocate_blocks_on_request(request)
        if request in self.waiting_requests:
            self.waiting_requests.remove(request)

        for bucket_idx in self.scheduled_requests.buckets.keys():
            if request in self.scheduled_requests.buckets[bucket_idx]:
                self.scheduled_requests.buckets[bucket_idx].remove(request)

    def is_empty(self):
        return not self.waiting_requests and len(self.scheduled_requests) == 0
    
    def schedule(self):
        return []


class SSLOSimulator:
    def __init__(self, args):
        self.args = args
        self.latency_data = self.load_latency_data(args.latency_data_path)

        self.scheduler = SSLOScheduler(args)

    def load_latency_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)


