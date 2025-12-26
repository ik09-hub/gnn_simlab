
import os
import json
import torch

from pipeline.tasks import TASK_REGISTRY
from pipeline.models import MODEL_REGISTRY


def run_evaluation(run_dir: str, spec: dict) -> dict:
    
    data_path = os.path.join(run_dir, "data.pt")
    data = torch.load(data_path, weights_only=False)

   
    task_type = spec["task"]["type"]
    task = TASK_REGISTRY[task_type](spec["task"])

    labels = task.get_labels(data)
    masks = task.get_masks(data)
    metrics = task.get_metrics()

    
    model_spec = spec["model"]
    model_type = model_spec["type"]

    ModelClass = MODEL_REGISTRY[model_type]
    model = ModelClass(
        in_channels=data.x.size(1),
        hidden_dim=model_spec.get("hidden_dim", 32)
    )

    model_path = os.path.join(run_dir, "model.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    #Evaluate on test only 
    results = {}

    with torch.no_grad():
        outputs = model(data)
        test_mask = masks["test"]

        for name, metric_fn in metrics.items():
            results[f"test_{name}"] = metric_fn(
                outputs[test_mask],
                labels[test_mask]
            )

    #Save
    eval_path = os.path.join(run_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation results:", results)

    return results
