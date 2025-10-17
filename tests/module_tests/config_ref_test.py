import torch

from transformers import AutoConfig


class ObjectInConfig:
    def __init__(self, number):
        self.number = number

class Module(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self):
        print(self.config.object_in_config.number)




if __name__ == "__main__":
    config = AutoConfig.from_pretrained("Qwen/Qwen2-72B-Instruct")

    num_objects = 10

    obj_list = [ObjectInConfig(i) for i in range(num_objects)]
    
    setattr(config, "object_in_config", None)
    module = Module(config)
    
    for obj in obj_list:
        setattr(config, "object_in_config", obj)
        module.forward()





