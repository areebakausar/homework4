import torch
import torch.nn as nn
from pathlib import Path

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class MLPPlanner(nn.Module):
    def __init__(
        self, 
        n_track: int = 10, 
        n_waypoints: int = 3, 
        hidden_dim: int = 256, 
        num_layers: int = 4, 
        dropout: int = 1e-9,
        **kwargs,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        
        input_size = n_track * 4
        output_size = n_waypoints * 2

        layers = []
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, track_left=None, track_right=None, track=None, *args, **kwargs) -> torch.Tensor:
        batch_size = track_left.shape[0]
        track_left_flat = track_left.reshape(batch_size, -1) 
        track_right_flat = track_right.reshape(batch_size, -1)
        track_flat = torch.cat([track_left_flat, track_right_flat], dim=1)
        
        
        output = self.model(track_flat)
        
        waypoints = output.reshape(batch_size, self.n_waypoints, 2) 
        
        return waypoints

class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        raise NotImplementedError

class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        num_layers: int = 2,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints
        self.height = 96
        self.width = 128

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        cnn_layers = []
        c1 = 3 # input channels for RGB image
        for c2 in [32, 64, 128]:
            cnn_layers.append(nn.Conv2d(c1, c2, kernel_size=3, padding=1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d(2))
            c1 = c2
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Build fully connected layers with specified hidden_dim and num_layers
        fc_layers = []
        fc_input_dim = 128 * 12 * 16
        
        for i in range(num_layers - 1):
            fc_layers.append(nn.Linear(fc_input_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            fc_input_dim = hidden_dim
        
        # Final output layer
        fc_layers.append(nn.Linear(fc_input_dim, n_waypoints * 2))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x.view(-1, self.n_waypoints, 2)

MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}

def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
