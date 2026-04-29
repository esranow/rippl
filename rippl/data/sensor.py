"""
Multi-fidelity data ingestion and PDE loss fusion.
Fuses: high-fidelity sparse sensor data + low-fidelity PDE physics.
"""
import torch
import pandas as pd
import numpy as np

class SensorDataset:
    """
    Loads and preprocesses sensor data for PINN training.
    Supports CSV, numpy arrays, torch tensors.
    """
    def __init__(self, coords: torch.Tensor,
                 fields: dict,
                 fidelity: float = 1.0,
                 noise_std: float = 0.0):
        self.coords = coords
        self.fields = fields
        self.fidelity = fidelity
        self.noise_std = noise_std

    @classmethod
    def from_csv(cls, path: str, coord_cols: list,
                 field_cols: dict, fidelity: float = 1.0,
                 noise_std: float = 0.0,
                 filter_outliers: bool = True) -> 'SensorDataset':
        df = pd.read_csv(path)
        if filter_outliers:
            df = cls._iqr_filter(df, list(field_cols.values()))
        coords = torch.tensor(df[coord_cols].values, dtype=torch.float32)
        fields = {
            k: torch.tensor(df[v].values[:, None], dtype=torch.float32)
            for k, v in field_cols.items()
        }
        return cls(coords, fields, fidelity, noise_std)

    @classmethod
    def from_numpy(cls, coords: np.ndarray, fields: dict,
                   **kwargs) -> 'SensorDataset':
        coords_t = torch.tensor(coords, dtype=torch.float32)
        fields_t = {k: torch.tensor(v, dtype=torch.float32) for k, v in fields.items()}
        return cls(coords_t, fields_t, **kwargs)

    @staticmethod
    def _iqr_filter(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def data_loss(self, model, device: str = "cpu") -> torch.Tensor:
        """
        Weighted MSE loss between model predictions and sensor values.
        """
        coords = self.coords.to(device)
        u_pred = model(coords)
        if not isinstance(u_pred, dict):
            u_pred = {"u": u_pred}
            
        weight = self.fidelity / (self.noise_std**2 + 1e-8)
        loss = torch.tensor(0.0, device=device)
        for f, target in self.fields.items():
            loss = loss + torch.mean((u_pred[f] - target.to(device))**2)
            
        return weight * loss

    def split(self, train_frac: float = 0.8) -> tuple:
        N = len(self)
        N_train = int(N * train_frac)
        indices = torch.randperm(N)
        train_idx = indices[:N_train]
        val_idx = indices[N_train:]
        
        train_ds = SensorDataset(self.coords[train_idx], 
                                 {k: v[train_idx] for k, v in self.fields.items()},
                                 self.fidelity, self.noise_std)
        val_ds = SensorDataset(self.coords[val_idx], 
                               {k: v[val_idx] for k, v in self.fields.items()},
                               self.fidelity, self.noise_std)
        return train_ds, val_ds

    def __len__(self): return self.coords.shape[0]


class MultiFidelityFusion:
    """
    Fuses multiple SensorDataset instances at different fidelity levels
    with PDE physics loss.
    """
    def __init__(self, datasets: list,
                 physics_weight: float = 1.0,
                 auto_balance: bool = True):
        self.datasets = datasets
        self.w_physics = physics_weight
        self.auto_balance = auto_balance

    def total_data_loss(self, model, device="cpu") -> torch.Tensor:
        if self.auto_balance:
            total_fidelity = sum(d.fidelity for d in self.datasets)
            return sum(
                (d.fidelity / (total_fidelity + 1e-8)) * d.data_loss(model, device)
                for d in self.datasets
            )
        return sum(d.data_loss(model, device) for d in self.datasets)

    def fusion_loss(self, model, physics_loss: torch.Tensor,
                    device="cpu") -> torch.Tensor:
        return self.total_data_loss(model, device) + self.w_physics * physics_loss
