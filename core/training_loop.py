"""Training loop for entropic unification of quantum and geometric systems."""

import torch
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

from .quantum_engine import QuantumEngine
from .geometry_engine import GeometryEngine
from .entropy_module import EntropyEngine

class EntropicUnificationTrainer:
    """Coordinates the training between quantum and geometric systems."""
    
    def __init__(self, config: Dict):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize components
        self.quantum_engine = QuantumEngine(config['quantum']).to(self.device)
        self.geometry_engine = GeometryEngine(**config['geometry']).to(self.device)
        self.entropy_engine = EntropyEngine(epsilon=1e-10)
        
        # Optimizers
        self.quantum_optimizer = optim.Adam(
            self.quantum_engine.parameters(),
            lr=config['training'].get('quantum_lr', 1e-3)
        )
        self.geometry_optimizer = optim.Adam(
            self.geometry_engine.parameters(),
            lr=config['training'].get('geometry_lr', 1e-4)
        )
        
        # Learning rate schedulers
        self.quantum_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.quantum_optimizer, 'min', patience=5, factor=0.5
        )
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'total_loss': [],
            'quantum_loss': [],
            'geometry_loss': [],
            'entropy': [],
            'curvature': []
        }
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'runs/run_') + datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def compute_quantum_state(self, inputs: torch.Tensor) -> torch.Tensor:
        """Compute quantum states from input parameters."""
        return self.quantum_engine(inputs)
    
    def compute_entropy(self, states: torch.Tensor) -> torch.Tensor:
        """Compute entanglement entropy for quantum states."""
        return torch.stack([self.entropy_engine.compute_entanglement_entropy(state) 
                          for state in states])
    
    def compute_geometry_loss(self, entropy: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute geometry-related losses from entropy distribution."""
        # Reshape entropy to match geometry engine's expected input
        entropy_field = entropy.view(-1, 1, 1).expand(-1, self.geometry_engine.lattice_size, 1)
        
        # Compute curvature from entropy
        curvature = self.geometry_engine.entropy_to_curvature(entropy_field)
        
        # Compute various curvature measures
        ricci = self.geometry_engine.compute_ricci_tensor()
        ricci_scalar = self.geometry_engine.compute_ricci_scalar()
        
        # Compute Einstein tensor (for energy-momentum tensor)
        einstein = self.geometry_engine.compute_einstein_tensor()
        
        # Compute losses
        loss_ricci = torch.mean(ricci ** 2)  # Encourage Ricci flatness
        loss_einstein = torch.mean(einstein ** 2)  # Minimize Einstein tensor
        
        # Total geometry loss
        loss = (
            self.config['training'].get('ricci_weight', 1.0) * loss_ricci +
            self.config['training'].get('einstein_weight', 1.0) * loss_einstein
        )
        
        metrics = {
            'ricci_loss': loss_ricci.item(),
            'einstein_loss': loss_einstein.item(),
            'avg_curvature': torch.mean(torch.abs(curvature)).item(),
            'avg_ricci_scalar': torch.mean(torch.abs(ricci_scalar)).item()
        }
        
        return loss, metrics
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        # Move batch to device
        inputs = batch['inputs'].to(self.device)
        targets = batch.get('targets')
        if targets is not None:
            targets = targets.to(self.device)
        
        # Forward pass through quantum system
        states = self.compute_quantum_state(inputs)
        entropy = self.compute_entropy(states)
        
        # Compute quantum loss (example: match target entropy if provided)
        if targets is not None:
            quantum_loss = torch.mean((entropy - targets) ** 2)
        else:
            # Default: maximize entanglement entropy
            quantum_loss = -torch.mean(entropy)
        
        # Update quantum parameters
        self.quantum_optimizer.zero_grad()
        quantum_loss.backward(retain_graph=True)
        self.quantum_optimizer.step()
        
        # Compute geometry loss
        geometry_loss, geometry_metrics = self.compute_geometry_loss(entropy.detach())
        
        # Update geometry parameters
        self.geometry_optimizer.zero_grad()
        geometry_loss.backward()
        self.geometry_optimizer.step()
        
        # Update learning rates
        self.quantum_scheduler.step(quantum_loss)
        
        # Compute total loss
        total_loss = quantum_loss + geometry_loss
        
        # Update metrics
        metrics = {
            'total_loss': total_loss.item(),
            'quantum_loss': quantum_loss.item(),
            'geometry_loss': geometry_loss.item(),
            'avg_entropy': torch.mean(entropy).item(),
            **geometry_metrics
        }
        
        return metrics
    
    def train(self, train_loader, val_loader=None, num_epochs: int = 100):
        """Run the main training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training phase
            self.quantum_engine.train()
            self.geometry_engine.train()
            
            train_metrics = {k: [] for k in self.history.keys()}
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as t:
                for batch in t:
                    metrics = self.train_step(batch)
                    
                    # Update progress bar
                    t.set_postfix(**{k: f'{v:.4f}' for k, v in metrics.items()})
                    
                    # Accumulate metrics
                    for k in train_metrics:
                        if k in metrics:
                            train_metrics[k].append(metrics[k])
            
            # Compute epoch metrics
            train_metrics = {k: np.mean(v) for k, v in train_metrics.items() if v}
            
            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                # Save best model
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    self.save_checkpoint(best=True)
            
            # Update history
            self._update_history(train_metrics, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config['training'].get('save_interval', 10) == 0:
                self.save_checkpoint()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Run validation."""
        self.quantum_engine.eval()
        self.geometry_engine.eval()
        
        val_metrics = {k: [] for k in self.history.keys()}
        
        for batch in val_loader:
            inputs = batch['inputs'].to(self.device)
            targets = batch.get('targets')
            if targets is not None:
                targets = targets.to(self.device)
            
            # Forward pass
            states = self.compute_quantum_state(inputs)
            entropy = self.compute_entropy(states)
            
            # Compute quantum loss
            if targets is not None:
                quantum_loss = torch.mean((entropy - targets) ** 2)
            else:
                quantum_loss = -torch.mean(entropy)
            
            # Compute geometry loss
            geometry_loss, geometry_metrics = self.compute_geometry_loss(entropy)
            
            # Compute metrics
            metrics = {
                'total_loss': (quantum_loss + geometry_loss).item(),
                'quantum_loss': quantum_loss.item(),
                'geometry_loss': geometry_loss.item(),
                'avg_entropy': torch.mean(entropy).item(),
                **geometry_metrics
            }
            
            # Accumulate metrics
            for k in val_metrics:
                if k in metrics:
                    val_metrics[k].append(metrics[k])
        
        # Average metrics
        return {k: np.mean(v) for k, v in val_metrics.items() if v}
    
    def _update_history(self, train_metrics: Dict, val_metrics: Dict):
        """Update training history with metrics from current epoch."""
        for k in self.history:
            if k in train_metrics:
                self.history[k].append(train_metrics[k])
    
    def _log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to console and file."""
        # Console logging
        log_str = f"Epoch {epoch+1}:"
        
        # Training metrics
        log_str += "\n  Training:   "
        log_str += ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        
        # Validation metrics
        if val_metrics:
            log_str += "\n  Validation: "
            log_str += ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
        
        print(log_str)
        
        # Save metrics to file
        with open(self.output_dir / 'metrics.txt', 'a') as f:
            f.write(log_str + '\n')
    
    def save_checkpoint(self, best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'quantum_state_dict': self.quantum_engine.state_dict(),
            'geometry_state_dict': self.geometry_engine.state_dict(),
            'quantum_optimizer': self.quantum_optimizer.state_dict(),
            'geometry_optimizer': self.geometry_optimizer.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        if best:
            torch.save(checkpoint, self.output_dir / 'best_model.pt')
        else:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{self.epoch}.pt')
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, config: Optional[Dict] = None):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Use config from checkpoint if not provided
        if config is None:
            config = checkpoint['config']
        
        # Initialize trainer
        trainer = cls(config)
        
        # Load state dicts
        trainer.quantum_engine.load_state_dict(checkpoint['quantum_state_dict'])
        trainer.geometry_engine.load_state_dict(checkpoint['geometry_state_dict'])
        trainer.quantum_optimizer.load_state_dict(checkpoint['quantum_optimizer'])
        trainer.geometry_optimizer.load_state_dict(checkpoint['geometry_optimizer'])
        trainer.history = checkpoint['history']
        trainer.epoch = checkpoint['epoch']
        
        return trainer


def create_example_config() -> Dict:
    """Create an example configuration dictionary."""
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'output_dir': 'runs/example_run',
        
        'quantum': {
            'num_qubits': 4,
            'depth': 3,
            'device': 'default.qubit',
            'interface': 'torch'
        },
        
        'geometry': {
            'dimensions': 4,
            'lattice_size': 32,
            'initial_metric': 'minkowski',
            'dx': 0.1
        },
        
        'training': {
            'quantum_lr': 1e-3,
            'geometry_lr': 1e-4,
            'ricci_weight': 1.0,
            'einstein_weight': 1.0,
            'batch_size': 32,
            'num_epochs': 100,
            'save_interval': 10
        }
    }


if __name__ == "__main__":
    # Example usage
    config = create_example_config()
    
    # Create dummy data loaders (replace with real data)
    train_loader = [{'inputs': torch.randn(config['training']['batch_size'], 10)}] * 100
    val_loader = [{'inputs': torch.randn(config['training']['batch_size'], 10)}] * 20
    
    # Initialize and train
    trainer = EntropicUnificationTrainer(config)
    trainer.train(train_loader, val_loader, num_epochs=config['training']['num_epochs'])
