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

HOMEWORK_DIR = Path(__file__).resolve().parent

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 5e-4,
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

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data(
        "drive_data/train",
        transform_pipeline="default",
        return_dataloader=True,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
    )

    val_data = load_data(
        "drive_data/val",
        transform_pipeline="default",
        return_dataloader=True,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
    )

    loss_func = torch.nn.L1Loss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    best_val_lateral = float('inf')
    best_model_path = log_dir / f"best_{model_name}.th"
    
    patience = 50
    patience_counter = 0

    global_step = 0
    metrics = {
        "train_loss": [],
        "val_loss": [],
        "val_l1": [],
        "val_longitudinal": [],
        "val_lateral": [],
    }

    # training loop
    for epoch in range(num_epoch):
        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch in train_data:
            # Extract data from batch
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            waypoints_mask = batch["waypoints_mask"].to(device)

            pred_waypoints = model(track_left=track_left, track_right=track_right)

            masked_waypoints = waypoints * waypoints_mask[..., None]
            masked_pred = pred_waypoints * waypoints_mask[..., None]
            
            error = (pred_waypoints - waypoints).abs()
            error_masked = error * waypoints_mask[..., None]
            num_valid = waypoints_mask.sum().clamp(min=1.0)
            
            # Separate losses for longitudinal and lateral, with different weights
            lon_error = error_masked[..., 0].sum() / num_valid
            lat_error = error_masked[..., 1].sum() / num_valid
            loss_val = 0.3 * lon_error + 0.7 * lat_error


            optimizer.zero_grad(set_to_none=True)
            loss_val.backward()
            optimizer.step()

            global_step += 1
            metrics["train_loss"].append(loss_val.item())

        with torch.inference_mode():
            model.eval()

            metric = PlannerMetric()

            for batch in val_data:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                pred_waypoints = model(track_left=track_left, track_right=track_right)

                masked_waypoints = waypoints * waypoints_mask[..., None]
                masked_pred = pred_waypoints * waypoints_mask[..., None]
                
                lon_loss = loss_func(masked_pred[..., 0], masked_waypoints[..., 0])
                lat_loss = loss_func(masked_pred[..., 1], masked_waypoints[..., 1])
                loss_val = 0.3 * lon_loss + 0.7 * lat_loss
                metrics["val_loss"].append(loss_val.item())

                metric.add(pred_waypoints, waypoints, waypoints_mask)

            planner_metrics = metric.compute()
            metrics["val_l1"].append(planner_metrics["l1_error"])
            metrics["val_longitudinal"].append(planner_metrics["longitudinal_error"])
            metrics["val_lateral"].append(planner_metrics["lateral_error"])

        epoch_train_loss = torch.as_tensor(metrics["train_loss"]).mean()
        epoch_val_loss = torch.as_tensor(metrics["val_loss"]).mean()
        epoch_val_l1 = torch.as_tensor(metrics["val_l1"]).mean()
        epoch_val_longitudinal = torch.as_tensor(metrics["val_longitudinal"]).mean()
        epoch_val_lateral = torch.as_tensor(metrics["val_lateral"]).mean()

        logger.add_scalar("train_loss", epoch_train_loss, global_step)
        logger.add_scalar("val_loss", epoch_val_loss, global_step)
        logger.add_scalar("val_l1_error", epoch_val_l1, global_step)
        logger.add_scalar("val_longitudinal_error", epoch_val_longitudinal, global_step)
        logger.add_scalar("val_lateral_error", epoch_val_lateral, global_step)

        scheduler.step(epoch_val_loss)
        
        if epoch_val_lateral < best_val_lateral:
            best_val_lateral = epoch_val_lateral
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_longitudinal = {epoch_val_longitudinal:.3f})  (val_lateral={epoch_val_lateral:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: lateral error didn't improve for {patience} epochs")
                break

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={epoch_train_loss:.2f} "
                f"val_loss={epoch_val_loss:.2f} "
                f"val_l1_error={epoch_val_l1:.2f}"
            )
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"val_longitudinal_error={epoch_val_longitudinal:.2f} "
                f"val_lateral_error={epoch_val_lateral:.2f} "
            )

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(f"Loaded best model from {best_model_path}")
    
    save_model(model)

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")
