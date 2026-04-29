import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class OperatorDataset(Dataset):
    def __init__(self, input_functions: torch.Tensor, 
                 output_functions: torch.Tensor):
        self.input_functions = input_functions
        self.output_functions = output_functions
    
    def __len__(self):
        return self.input_functions.shape[0]
    
    def __getitem__(self, idx):
        return self.input_functions[idx], self.output_functions[idx]
    
    def get_batch(self, batch_size) -> tuple:
        # Simple random sampling for quick batches if needed
        indices = torch.randint(0, len(self), (batch_size,))
        return self.input_functions[indices], self.output_functions[indices]

class OperatorExperiment:
    def __init__(self, model, dataset, system=None):
        self.model = model
        self.dataset = dataset
        self.system = system # Optional for PI-Operator Learning
    
    @classmethod
    def from_flywheel(cls, flywheel):
        """Constructs OperatorExperiment from a trained flywheel's dataset."""
        return cls(flywheel.fno_model, flywheel.dataset_train, system=flywheel.system)
    
    def train(self, epochs=1000, lr=1e-3, physics_weight=0.0):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            total_data_loss = 0.0
            total_physics_loss = 0.0
            
            for a_batch, u_batch in loader:
                optimizer.zero_grad()
                
                # Data loss
                u_pred = self.model(a_batch)
                data_loss = F.mse_loss(u_pred, u_batch)
                
                # Physics loss
                physics_loss = torch.tensor(0.0, device=a_batch.device)
                if physics_weight > 0 and self.system:
                    # In PI-FNO, we often evaluate the residual on the predicted field
                    # u_pred: (B, N, F_out)
                    # We need coordinates to compute derivatives.
                    # Assuming a fixed grid for the FNO discretization.
                    # This part is simplified for the spec.
                    pass
                
                loss = data_loss + physics_weight * physics_loss
                loss.backward()
                optimizer.step()
                
                total_data_loss += data_loss.item()
                total_physics_loss += physics_loss.item()
            
            if (epoch + 1) % 100 == 0 or epoch == 0:
                msg = f"Epoch {epoch+1:4d} | Data Loss: {total_data_loss/len(loader):.6e}"
                if physics_weight > 0:
                    msg += f" | Physics Loss: {total_physics_loss/len(loader):.6e}"
                print(msg)
