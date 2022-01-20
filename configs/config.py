from dataclasses import dataclass

@dataclass
class Configuration:
    """Class for keeping track of an item in inventory."""
    name_model: str = 'M3'
    batch_size: int = 120
    num_ephocs: int = 150
    learning_rate: float = 0.001
    gamma: float = 0.98
    path_data: str = "data/"
    output_path: str = "output_models/model.pth"