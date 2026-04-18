"""
Usage:
    python3 -m homework.train_planner --your_args here
"""


import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.datasets.road_dataset import load_data
from homework.models import HOMEWORK_DIR, load_model, save_model
from homework.metrics import PlannerMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    weight_decay: float = 1e-4,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs for models
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data(
        "drive_data/train",
        transform_pipeline="default",
        return_dataloader=True,
        num_workers=2,
        batch_size=batch_size,
        shuffle=True,
    )

    val_data = load_data(
        "drive_data/val",
        transform_pipeline="default",
        return_dataloader=True,
        num_workers=2,
        batch_size=batch_size,
        shuffle=False,
    )

    # create loss function and optimizer
    loss_func = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    global_step = 0
    metrics = {"train_loss": [], "val_loss": [], "val_l1": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics: metrics[key].clear()

        model.train()

        for batch in train_data:
            # Extract data from batch
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            pred_waypoints = model(track_left=track_left, track_right=track_right)
            
            # Mask out invalid waypoints and compute loss
            masked_waypoints = waypoints * waypoints_mask[..., None]
            masked_pred = pred_waypoints * waypoints_mask[..., None]
            loss_val = loss_func(masked_pred, masked_waypoints)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            global_step += 1
            metrics["train_loss"].append(loss_val.item())

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()
            
            # Prepare metric collector
            metric = PlannerMetric()

            for batch in val_data:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                pred_waypoints = model(track_left=track_left, track_right=track_right)

                # Calculate loss
                masked_waypoints = waypoints * waypoints_mask[..., None]
                masked_pred = pred_waypoints * waypoints_mask[..., None]
                loss_val = loss_func(masked_pred, masked_waypoints)
                metrics["val_loss"].append(loss_val.item())
                
                # Add to metric
                metric.add(pred_waypoints, waypoints, waypoints_mask)
            
            # Get planner metrics
            planner_metrics = metric.compute()
            metrics["val_l1"].append(planner_metrics["l1_error"])
                

        # log average train and val metrics to tensorboard
        epoch_train_loss = torch.as_tensor(metrics["train_loss"]).mean()
        epoch_val_loss = torch.as_tensor(metrics["val_loss"]).mean()
        epoch_val_l1 = torch.as_tensor(metrics["val_l1"]).mean()

        logger.add_scalar("train_loss", epoch_train_loss, global_step)
        logger.add_scalar("val_loss", epoch_val_loss, global_step)
        logger.add_scalar("val_l1_error", epoch_val_l1, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={epoch_train_loss:.4f} "
                f"val_loss={epoch_val_loss:.4f} "
                f"val_l1_error={epoch_val_l1:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
