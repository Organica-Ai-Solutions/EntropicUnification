"""
Advanced optimization techniques for the EntropicUnification framework.

This module provides enhanced optimization algorithms including:
- ADAM optimizer
- Learning rate scheduling
- Momentum-based optimization
- Basin hopping for better local minima escape
- Early stopping based on validation metrics
"""

import numpy as np
import torch
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable

from .optimizer import EntropicOptimizer, OptimizerConfig


class OptimizerType(str, Enum):
    """Different optimizer algorithms."""
    SGD = "sgd"  # Standard stochastic gradient descent
    ADAM = "adam"  # Adaptive Moment Estimation
    RMSPROP = "rmsprop"  # Root Mean Square Propagation
    ADAGRAD = "adagrad"  # Adaptive Gradient Algorithm
    ADADELTA = "adadelta"  # Extension of Adagrad


class LRScheduleType(str, Enum):
    """Different learning rate scheduling strategies."""
    CONSTANT = "constant"  # Constant learning rate
    STEP = "step"  # Step decay
    EXPONENTIAL = "exponential"  # Exponential decay
    COSINE = "cosine"  # Cosine annealing
    REDUCE_ON_PLATEAU = "reduce_on_plateau"  # Reduce when loss plateaus


@dataclass
class AdvancedOptimizerConfig(OptimizerConfig):
    """Extended configuration for advanced optimization techniques."""
    optimizer_type: OptimizerType = OptimizerType.ADAM
    lr_schedule: LRScheduleType = LRScheduleType.COSINE
    
    # ADAM parameters
    beta1: float = 0.9  # Exponential decay rate for first moment
    beta2: float = 0.999  # Exponential decay rate for second moment
    epsilon: float = 1e-8  # Small constant for numerical stability
    
    # Learning rate scheduling parameters
    lr_decay_rate: float = 0.9  # For exponential and step decay
    lr_decay_steps: int = 20  # For step decay
    lr_min: float = 1e-6  # Minimum learning rate
    
    # Early stopping parameters
    early_stopping: bool = True
    patience: int = 10  # Number of iterations to wait for improvement
    min_delta: float = 1e-4  # Minimum change to qualify as improvement
    
    # Basin hopping parameters
    basin_hopping: bool = False
    temperature: float = 1.0  # Temperature parameter for basin hopping
    step_size: float = 0.5  # Step size for basin hopping


class AdvancedEntropicOptimizer(EntropicOptimizer):
    """Enhanced optimizer with advanced techniques for the EntropicUnification framework."""
    
    def __init__(self, quantum_engine, geometry_engine, entropy_module, 
                 coupling_layer, loss_functions, config: AdvancedOptimizerConfig):
        """Initialize the advanced optimizer."""
        super().__init__(quantum_engine, geometry_engine, entropy_module, 
                         coupling_layer, loss_functions, config)
        
        self.config = config
        self.step_count = 0
        
        # Initialize optimizer state
        self.m = None  # First moment estimate (momentum)
        self.v = None  # Second moment estimate (velocity)
        
        # Early stopping variables
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.stopped_early = False
        
        # Learning rate scheduling
        self.initial_lr = config.learning_rate
        self.current_lr = config.learning_rate
        
        # History tracking
        self.lr_history = []
    
    def _init_optimizer_state(self, metric_shape):
        """Initialize optimizer state variables based on metric shape."""
        if self.config.optimizer_type in [OptimizerType.ADAM, OptimizerType.RMSPROP]:
            self.m = torch.zeros_like(self.geometry.metric_field)
            self.v = torch.zeros_like(self.geometry.metric_field)
    
    def _update_learning_rate(self):
        """Update learning rate based on scheduling strategy."""
        if self.config.lr_schedule == LRScheduleType.CONSTANT:
            return
        
        if self.config.lr_schedule == LRScheduleType.STEP:
            # Step decay: lr = initial_lr * decay_rate^(step / decay_steps)
            decay_factor = self.config.lr_decay_rate ** (self.step_count / self.config.lr_decay_steps)
            self.current_lr = max(self.initial_lr * decay_factor, self.config.lr_min)
        
        elif self.config.lr_schedule == LRScheduleType.EXPONENTIAL:
            # Exponential decay: lr = initial_lr * decay_rate^step
            decay_factor = self.config.lr_decay_rate ** self.step_count
            self.current_lr = max(self.initial_lr * decay_factor, self.config.lr_min)
        
        elif self.config.lr_schedule == LRScheduleType.COSINE:
            # Cosine annealing: lr = lr_min + 0.5 * (initial_lr - lr_min) * (1 + cos(step / steps * pi))
            if self.step_count >= self.config.steps:
                self.current_lr = self.config.lr_min
            else:
                cosine_factor = 0.5 * (1 + np.cos(np.pi * self.step_count / self.config.steps))
                self.current_lr = self.config.lr_min + (self.initial_lr - self.config.lr_min) * cosine_factor
        
        elif self.config.lr_schedule == LRScheduleType.REDUCE_ON_PLATEAU:
            # This is handled separately in the train method when loss plateaus
            pass
        
        self.lr_history.append(self.current_lr)
    
    def _apply_adam_update(self, gradient):
        """Apply ADAM optimization update."""
        if self.m is None or self.v is None:
            self._init_optimizer_state(gradient.shape)
        
        # Update biased first moment estimate
        self.m = self.config.beta1 * self.m + (1 - self.config.beta1) * gradient
        
        # Update biased second raw moment estimate
        self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * gradient**2
        
        # Bias correction
        m_hat = self.m / (1 - self.config.beta1**(self.step_count + 1))
        v_hat = self.v / (1 - self.config.beta2**(self.step_count + 1))
        
        # Apply update
        update = self.current_lr * m_hat / (torch.sqrt(v_hat) + self.config.epsilon)
        return update
    
    def _apply_rmsprop_update(self, gradient):
        """Apply RMSProp optimization update."""
        if self.v is None:
            self._init_optimizer_state(gradient.shape)
        
        # Update accumulated squared gradient
        self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * gradient**2
        
        # Apply update
        update = self.current_lr * gradient / (torch.sqrt(self.v) + self.config.epsilon)
        return update
    
    def _apply_sgd_update(self, gradient):
        """Apply standard SGD update with optional momentum."""
        if self.m is None:
            self._init_optimizer_state(gradient.shape)
        
        # Apply momentum if beta1 > 0
        if self.config.beta1 > 0:
            self.m = self.config.beta1 * self.m + gradient
            update = self.current_lr * self.m
        else:
            update = self.current_lr * gradient
        
        return update
    
    def _check_early_stopping(self, loss):
        """Check if early stopping criteria are met."""
        if not self.config.early_stopping:
            return False
        
        if loss < self.best_loss - self.config.min_delta:
            # We found a better loss
            self.best_loss = loss
            self.patience_counter = 0
            return False
        else:
            # No improvement
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                self.stopped_early = True
                return True
            return False
    
    def _apply_basin_hopping(self, metric, loss):
        """Apply basin hopping to escape local minima."""
        if not self.config.basin_hopping:
            return metric, loss
        
        # Only apply basin hopping occasionally
        if np.random.random() > 0.1:  # 10% chance
            return metric, loss
        
        # Create a perturbed metric
        perturbed_metric = metric + torch.randn_like(metric) * self.config.step_size
        
        # Evaluate the loss with the perturbed metric
        original_metric = self.geometry.metric_field
        self.geometry.metric_field = perturbed_metric
        
        # Recompute all tensors with the new metric
        self.geometry.clear_cache()
        
        # Compute the loss with the perturbed metric
        # We need to recompute the coupling terms with the new metric
        state = self.quantum.state
        partition = list(range(self.quantum.num_qubits // 2))  # Default partition
        terms = self.coupling.compute_coupling_terms(state, partition)
        perturbed_loss = self.loss.total_loss(terms, None, None)
        
        # Metropolis acceptance criterion
        delta_loss = perturbed_loss - loss
        if delta_loss < 0 or np.random.random() < np.exp(-delta_loss / self.config.temperature):
            # Accept the perturbation
            return perturbed_metric, perturbed_loss
        else:
            # Reject the perturbation
            self.geometry.metric_field = original_metric
            self.geometry.clear_cache()
            return metric, loss
    
    def optimization_step(self, state, partition, target_gradient=None, weights=None):
        """Perform a single optimization step with advanced techniques."""
        # Compute coupling terms
        terms = self.coupling.compute_coupling_terms(state, partition)
        
        # Compute loss and its gradient
        losses = self.loss.total_loss(terms, target_gradient, weights)
        total_loss = losses["total"]
        
        # Compute gradient of loss with respect to metric
        metric_gradient = self.compute_metric_gradient(state, partition, target_gradient, weights)
        
        # Clip gradient if needed
        if self.config.metric_grad_clip > 0:
            metric_gradient = torch.clamp(
                metric_gradient, 
                -self.config.metric_grad_clip, 
                self.config.metric_grad_clip
            )
        
        # Update learning rate
        self._update_learning_rate()
        
        # Apply optimizer-specific update
        if self.config.optimizer_type == OptimizerType.ADAM:
            update = self._apply_adam_update(metric_gradient)
        elif self.config.optimizer_type == OptimizerType.RMSPROP:
            update = self._apply_rmsprop_update(metric_gradient)
        else:
            update = self._apply_sgd_update(metric_gradient)
        
        # Update metric
        self.geometry.update_metric(-update)  # Negative because we're minimizing
        
        # Apply basin hopping if enabled
        if self.config.basin_hopping:
            self.geometry.metric_field, new_loss = self._apply_basin_hopping(
                self.geometry.metric_field, total_loss
            )
            if new_loss != total_loss:
                # Recompute losses with the new metric
                terms = self.coupling.compute_coupling_terms(state, partition)
                losses = self.loss.total_loss(terms, target_gradient, weights)
                total_loss = losses["total"]
        
        # Increment step counter
        self.step_count += 1
        
        return losses
    
    def train(self, parameters, times, partition, weights=None, initial_states=None, target_gradient=None):
        """Run the full optimization loop with advanced techniques."""
        # Initialize state
        if initial_states is None:
            initial_states = torch.zeros((len(times), 2**self.quantum.num_qubits), dtype=torch.complex128)
            initial_states[:, 0] = 1.0
        
        # Evolve quantum state
        states = self.quantum.time_evolve_batch(parameters, times, initial_states)
        
        # Select active state
        state = states[self.active_index]
        self.quantum.state = state
        
        # Initialize history
        history = {
            "total_loss": [],
            "einstein_loss": [],
            "entropy_loss": [],
            "regularity_loss": [],
            "learning_rate": []
        }
        
        # Initialize best state tracking
        best_loss = float('inf')
        best_metric = None
        best_step = 0
        
        # Main optimization loop
        from tqdm import tqdm
        pbar = tqdm(range(self.config.steps), desc="Entropic optimization")
        
        for step in pbar:
            # Perform optimization step
            losses = self.optimization_step(state, partition, target_gradient, weights)
            
            # Update history
            history["total_loss"].append(losses["total"].item())
            history["einstein_loss"].append(losses["einstein"].item())
            history["entropy_loss"].append(losses["entropy"].item())
            if "regularity" in losses:
                history["regularity_loss"].append(losses["regularity"].item())
            else:
                history["regularity_loss"].append(0.0)
            history["learning_rate"].append(self.current_lr)
            
            # Track best state
            if losses["total"].item() < best_loss:
                best_loss = losses["total"].item()
                best_metric = self.geometry.metric_field.clone()
                best_step = step
            
            # Update progress bar
            pbar.set_description(f"Entropic optimization: Loss={losses['total']:.6f}, LR={self.current_lr:.6f}")
            
            # Check for early stopping
            if self._check_early_stopping(losses["total"].item()):
                pbar.set_description(f"Entropic optimization: Converged after {step+1} steps!")
                break
            
            # Learning rate scheduling for reduce_on_plateau
            if (self.config.lr_schedule == LRScheduleType.REDUCE_ON_PLATEAU and 
                self.patience_counter > self.config.patience // 2):
                self.current_lr *= self.config.lr_decay_rate
                self.patience_counter = 0
                pbar.set_description(f"Entropic optimization: Reducing LR to {self.current_lr:.6f}")
        
        # Restore best metric if needed
        if self.config.early_stopping and best_metric is not None:
            self.geometry.metric_field = best_metric
            self.geometry.clear_cache()
        
        # Prepare results
        results = {
            "final_state": state,
            "history": history,
            "best_loss": best_loss,
            "best_step": best_step,
            "converged": self.stopped_early,
            "final_metric": self.geometry.metric_field.clone()
        }
        
        return results
    
    def analyze_convergence(self, history=None):
        """Analyze the convergence behavior of the optimization."""
        if history is None:
            if not hasattr(self, 'last_history'):
                return {
                    "converged": False,
                    "final_loss": 0.0,
                    "best_loss": 0.0,
                    "convergence_rate": 0.0
                }
            history = self.last_history
        
        losses = history["total_loss"]
        if not losses:
            return {
                "converged": False,
                "final_loss": 0.0,
                "best_loss": 0.0,
                "convergence_rate": 0.0
            }
        
        # Calculate convergence metrics
        final_loss = losses[-1]
        best_loss = min(losses)
        best_idx = losses.index(best_loss)
        
        # Estimate convergence rate from last 25% of training
        start_idx = max(0, len(losses) // 4 * 3)
        if start_idx < len(losses) - 1:
            recent_losses = losses[start_idx:]
            if len(recent_losses) > 1:
                convergence_rate = (recent_losses[0] - recent_losses[-1]) / len(recent_losses)
            else:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
        
        # Determine if converged based on several criteria
        converged = False
        if len(losses) > 10:
            # Check if loss is stable in the last 10 iterations
            recent_std = np.std(losses[-10:])
            recent_mean = np.mean(losses[-10:])
            
            # Converged if standard deviation is small relative to the mean
            if recent_std / (recent_mean + 1e-10) < 0.01:
                converged = True
            
            # Converged if we stopped early
            if self.stopped_early:
                converged = True
            
            # Not converged if we're still making good progress
            if convergence_rate > 0.001:
                converged = False
        
        return {
            "converged": converged,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "best_step": best_idx,
            "convergence_rate": convergence_rate
        }
    
    def plot_learning_rate_schedule(self, steps=None):
        """Plot the learning rate schedule."""
        import matplotlib.pyplot as plt
        
        if steps is None:
            steps = self.config.steps
        
        # Generate learning rate schedule
        lr_schedule = []
        original_step = self.step_count
        self.step_count = 0
        self.current_lr = self.initial_lr
        
        for step in range(steps):
            self.step_count = step
            self._update_learning_rate()
            lr_schedule.append(self.current_lr)
        
        # Reset state
        self.step_count = original_step
        self._update_learning_rate()
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(steps), lr_schedule)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title(f'Learning Rate Schedule: {self.config.lr_schedule.value}')
        plt.grid(True)
        
        return plt.gcf()
