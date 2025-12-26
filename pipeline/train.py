import os
import torch
import json
from pipeline.tasks import TASK_REGISTRY
from pipeline.models import MODEL_REGISTRY

def run_training(run_dir: str, spec: dict) -> str:

    data_path = os.path.join(run_dir, "data.pt")
    data = torch.load(data_path, weights_only = False)

    task_type = spec["task"]["type"]
    task = TASK_REGISTRY[task_type](spec["task"])

    labels = task.get_labels(data)
    masks = task.get_masks(data)
    loss_fn = task.get_loss()
    metrics = task.get_metrics()

    model_spec = spec["model"]
    model_type = model_spec["type"]

    ModelClass = MODEL_REGISTRY[model_type]
    model = ModelClass(
        in_channels=data.x.size(1),
        hidden_dim=model_spec.get("hidden_dim", 32)
    )

    # -------------------------
    # Optimizer & config
    # -------------------------
    train_spec = spec.get("training", {})
    lr = train_spec.get("lr", 0.01)
    epochs = train_spec.get("epochs", 50)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(data)
        loss = loss_fn(
            outputs[masks["train"]],
            labels[masks["train"]].float()
        )

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch:03d}] Loss: {loss.item():.4f}")

    # -------------------------
    # Evaluation
    # -------------------------
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        results = {}

        for split, mask in masks.items():
            if mask.sum() == 0:
                continue
            for name, metric_fn in metrics.items():
                results[f"{split}_{name}"] = metric_fn(
                    outputs[mask],
                    labels[mask]
                )

    # -------------------------
    # Save artifacts
    # -------------------------
    model_path = os.path.join(run_dir, "model.pt")
    metrics_path = os.path.join(run_dir, "metrics.json")

    torch.save(model.state_dict(), model_path)
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    return model_path





