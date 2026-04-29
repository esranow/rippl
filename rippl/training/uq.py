"""
rippl.training.uq — Uncertainty Quantification for PINNs and Operators.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union
from rippl.core.experiment import Experiment

class MCDropoutWrapper(nn.Module):
    """
    Monte Carlo Dropout Wrapper.
    Inserts Dropout layers after every non-output linear layer.
    """
    def __init__(self, model: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.base_model = model
        self.dropout_rate = dropout_rate
        self._inject_dropout(self.base_model, dropout_rate)

    def _inject_dropout(self, model, rate):
        """
        Inserts dropout after every non-output linear layer.
        """
        linears = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linears.append((name, module))
        
        if len(linears) <= 1: return
        
        # The last linear is the output layer, don't add dropout after it
        output_layer_name, _ = linears[-1]
        
        # Replace non-output linear layers with (Linear -> Dropout)
        for name, module in linears[:-1]:
            # Navigate to the parent module
            parts = name.split('.')
            curr = model
            for part in parts[:-1]:
                curr = getattr(curr, part)
            
            # Create a wrapper
            class LinearDropout(nn.Module):
                def __init__(self, lin, p):
                    super().__init__()
                    self.lin = lin
                    self.dropout = nn.Dropout(p=p)
                def forward(self, x):
                    return self.dropout(self.lin(x))
            
            setattr(curr, parts[-1], LinearDropout(module, rate))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.base_model(coords)

    def predict_with_uncertainty(self, coords: torch.Tensor, n_samples: int = 50, device: str = "cpu") -> Dict[str, torch.Tensor]:
        self.to(device)
        coords = coords.to(device)
        self.train() # Enable dropout
        
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.base_model(coords)
                samples.append(out)
        
        samples_tensor = torch.stack(samples) # (n_samples, N, out)
        return {
            "mean": samples_tensor.mean(dim=0),
            "std": samples_tensor.std(dim=0),
            "samples": samples_tensor
        }

class DeepEnsemble:
    def __init__(self, models: List[nn.Module]):
        self.models = models

    def predict_with_uncertainty(self, coords: torch.Tensor, device: str = "cpu") -> Dict[str, torch.Tensor]:
        device_torch = torch.device(device)
        coords = coords.to(device_torch)
        
        samples = []
        with torch.no_grad():
            for model in self.models:
                model.to(device_torch)
                model.eval()
                out = model(coords)
                samples.append(out)
        
        samples_tensor = torch.stack(samples)
        return {
            "mean": samples_tensor.mean(dim=0),
            "std": samples_tensor.std(dim=0),
            "samples": samples_tensor
        }

    def __len__(self):
        return len(self.models)

class UncertaintyQuantifier:
    def __init__(self, model_or_ensemble, method: str = "mc_dropout", n_samples: int = 50):
        self.model = model_or_ensemble
        self.method = method
        self.n_samples = n_samples

    def predict(self, coords: torch.Tensor, device: str = "cpu") -> Dict[str, torch.Tensor]:
        if self.method == "mc_dropout":
            return self.model.predict_with_uncertainty(coords, n_samples=self.n_samples, device=device)
        return self.model.predict_with_uncertainty(coords, device=device)

    def confidence_interval(self, coords: torch.Tensor, alpha: float = 0.05, device: str = "cpu") -> Dict[str, torch.Tensor]:
        res = self.predict(coords, device=device)
        mean, std = res["mean"], res["std"]
        
        # Simple z-score approximation (1.96 for alpha=0.05)
        # For more precision, we'd use scipy.stats.norm.ppf(1 - alpha/2)
        try:
            from scipy import stats
            z = stats.norm.ppf(1 - alpha / 2)
        except ImportError:
            # Fallback for common alpha values
            if abs(alpha - 0.05) < 1e-3: z = 1.96
            elif abs(alpha - 0.01) < 1e-3: z = 2.576
            elif abs(alpha - 0.1) < 1e-3: z = 1.645
            else: z = 2.0 # conservative default
            
        return {
            "lower": mean - z * std,
            "upper": mean + z * std,
            "mean": mean
        }

    def epistemic_uncertainty(self, coords: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        return self.predict(coords, device=device)["std"]

    def high_uncertainty_regions(self, coords: torch.Tensor, threshold: float, device: str = "cpu") -> torch.Tensor:
        return self.epistemic_uncertainty(coords, device=device) > threshold

class ProbabilisticExperiment:
    def __init__(self, system, model, method: str = "mc_dropout", 
                 n_ensemble: int = 5, dropout_rate: float = 0.1, **experiment_kwargs):
        self.system = system
        self.method = method
        self.n_ensemble = n_ensemble
        self.dropout_rate = dropout_rate
        self.experiment_kwargs = experiment_kwargs
        
        if method == "mc_dropout":
            self.model = MCDropoutWrapper(model, dropout_rate=dropout_rate)
            self.experiment = Experiment(system, self.model, **experiment_kwargs)
        elif method == "ensemble":
            # We will store models and experiments after training
            self.base_model_template = model
            self.ensemble_models = []
            self.experiments = []
        else:
            raise ValueError(f"Unknown method {method}")

    def train(self, **kwargs):
        if self.method == "mc_dropout":
            return self.experiment.train(**kwargs)
        
        # Ensemble training
        import copy
        results = []
        for i in range(self.n_ensemble):
            # Re-instantiate model to ensure fresh weights
            # Deepcopying and then resetting weights if possible, 
            # but standard practice is to use the class to re-init.
            # Here we assume model can be cloned.
            new_model = copy.deepcopy(self.base_model_template)
            # Reset weights
            for layer in new_model.modules():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            
            # Fresh optimizer for each model
            if 'opt' in self.experiment_kwargs:
                # Re-create optimizer for the new model
                opt_type = type(self.experiment_kwargs['opt'])
                opt_kwargs = {k: v for k, v in self.experiment_kwargs['opt'].defaults.items()}
                new_opt = opt_type(new_model.parameters(), **opt_kwargs)
            else:
                # Fallback Adam
                new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
            
            kwargs_copy = copy.deepcopy(self.experiment_kwargs)
            kwargs_copy['opt'] = new_opt
            
            exp = Experiment(self.system, new_model, **kwargs_copy)
            res = exp.train(**kwargs)
            self.experiments.append(exp)
            self.ensemble_models.append(new_model)
            results.append(res)
        return results

    def predict_with_uncertainty(self, coords: torch.Tensor, n_samples: int = 50):
        if self.method == "mc_dropout":
            return self.model.predict_with_uncertainty(coords, n_samples=n_samples)
        
        ensemble = DeepEnsemble(self.ensemble_models)
        return ensemble.predict_with_uncertainty(coords)

    def uncertainty_report(self, coords: torch.Tensor, device: str = "cpu") -> Dict[str, float]:
        res = self.predict_with_uncertainty(coords)
        std = res["std"]
        return {
            "method": self.method,
            "mean_std": std.mean().item(),
            "max_std": std.max().item(),
            "high_uncertainty_fraction": (std > 0.1 * res["mean"].abs().mean()).float().mean().item()
        }
