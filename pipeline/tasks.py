from typing import Dict
import torch
import torch.nn.functional as F

"""
Task Interface: basic methods for each task
"""

class Task:
    def __init__(self, task_spec: Dict):
        self.task_spec = task_spec
    
    def get_labels(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def get_masks(self, data) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def get_loss(self, data):
        raise NotImplementedError
    
    def get_metrics(self, data):
        raise NotImplementedError


## Node Classification Task

class NodeClassificationTask(Task):

    def get_labels(self, data):
        return data.y
    
    def get_masks(self, data):

        #if we already have the masks just use them
        if hasattr(data, "train_mask"):
            return {
                "train": data.train_mask,
                "val": data.val_mask,
                "test": data.test_mask,
            }
        
        num_nodes = data.num_nodes
        perm = torch.randperm(num_nodes)

        train_end = int(0.7 * num_nodes)
        val_end = int(0.85 * num_nodes)

        #0 out masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[perm[:train_end]] = True
        val_mask[perm[train_end:val_end]] = True
        test_mask[perm[val_end:]] = True

        #attach masks to dataset
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return {
            "train": train_mask,
            "val": val_mask,
            "test": test_mask,
        }
    
    def get_loss(self):
        label_type = self.task_spec.get("label_type", "binary")

        if label_type == "binary":
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.CrossEntropyLoss()
        
    def get_metrics(self):
        #helper function to calculate accuracy
        def accuracy(logits, labels):
            preds = (logits > 0).long()
            return (preds == labels).float().mean().item()

        return {
            "accuracy": accuracy
        }
    


"""
TASK REGISTRY: holds all the tasks we have 
"""

TASK_REGISTRY = {
    #we have only one for now but we can add more later 
    "node_classification": NodeClassificationTask,
}


