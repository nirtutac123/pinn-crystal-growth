import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time


class PINN:
    """
    Physics-Informed Neural Network base class
    """
    
    def __init__(self, model, pde, device=None):
        """
        Initialize PINN.
        
        Args:
            model: Neural network model
            pde: PDE object that defines the equation, boundary conditions, etc.
            device: Torch device (cpu or cuda)
        """
        self.model = model
        self.pde = pde
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
    def pde_loss(self, x):
        """
        Calculate PDE residual loss.
        
        Args:
            x: Collocation points
            
        Returns:
            Mean squared residual
        """
        residual = self.pde.compute_residual(self.model, x)
        return torch.mean(residual**2)
    
    def bc_ic_loss(self):
        """
        Calculate boundary and initial condition loss.
        
        Returns:
            Combined boundary and initial condition MSE loss
        """
        bc = self.pde.get_boundary_conditions()
        ic = self.pde.get_initial_conditions()
        
        bc_points, bc_values = bc['points'], bc['values']
        ic_points, ic_values = ic['points'], ic['values']
        
        # Predict values at boundary and initial points
        bc_pred = self.model(bc_points)
        ic_pred = self.model(ic_points)
        
        # Calculate mean squared error
        bc_mse = torch.mean((bc_pred - bc_values)**2)
        ic_mse = torch.mean((ic_pred - ic_values)**2)
        
        return bc_mse + ic_mse
    
    def total_loss(self, x):
        """
        Calculate total loss.
        
        Args:
            x: Collocation points
            
        Returns:
            Dictionary containing PDE loss, BC/IC loss, and total loss
        """
        pde_loss = self.pde_loss(x)
        bc_ic_loss = self.bc_ic_loss()
        total = pde_loss + bc_ic_loss
        
        return {
            'pde_loss': pde_loss,
            'bc_ic_loss': bc_ic_loss,
            'total_loss': total
        }
    
    def train(self, optimizer, n_collocation_points=10000, n_epochs=10000, log_interval=100):
        """
        Train the PINN model.
        
        Args:
            optimizer: PyTorch optimizer
            n_collocation_points: Number of collocation points for PDE residual
            n_epochs: Number of training epochs
            log_interval: How often to log progress
            
        Returns:
            Dictionary of training history
        """
        history = {'pde_loss': [], 'bc_ic_loss': [], 'total_loss': []}
        
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Generate collocation points
            x = self.pde.generate_collocation_points(n_collocation_points)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute losses
            losses = self.total_loss(x)
            
            # Backward pass and optimizer step
            losses['total_loss'].backward()
            optimizer.step()
            
            # Record losses
            history['pde_loss'].append(losses['pde_loss'].item())
            history['bc_ic_loss'].append(losses['bc_ic_loss'].item())
            history['total_loss'].append(losses['total_loss'].item())
            
            # Log progress
            if epoch % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch}/{n_epochs}, "
                      f"PDE Loss: {losses['pde_loss'].item():.4e}, "
                      f"BC/IC Loss: {losses['bc_ic_loss'].item():.4e}, "
                      f"Total Loss: {losses['total_loss'].item():.4e}, "
                      f"Time: {elapsed_time:.2f}s")
                
        return history


class GNPINN(PINN):
    """
    Gradient-Normalized Physics-Informed Neural Network
    """
    
    def train(self, optimizer, n_collocation_points=10000, n_epochs=10000, log_interval=100):
        """
        Train the GNPINN model using gradient normalization.
        
        Args:
            optimizer: PyTorch optimizer
            n_collocation_points: Number of collocation points for PDE residual
            n_epochs: Number of training epochs
            log_interval: How often to log progress
            
        Returns:
            Dictionary of training history
        """
        history = {'pde_loss': [], 'bc_ic_loss': [], 'total_loss': []}
        
        for epoch in range(n_epochs):
            start_time = time.time()
            self.model.train()
            
            # Generate collocation points
            x = self.pde.generate_collocation_points(n_collocation_points)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Calculate PDE loss
            pde_loss = self.pde_loss(x)
            
            # Calculate BC/IC loss
            bc_ic_loss = self.bc_ic_loss()
            
            # Total loss (not used for backpropagation in GN-PINN)
            total_loss = pde_loss + bc_ic_loss
            
            try:
                # --- Gradient Normalization ---
                # Get gradients for PDE loss
                pde_grads = torch.autograd.grad(pde_loss, self.model.parameters(), 
                                              retain_graph=True, create_graph=True, allow_unused=True)
                
                # Get gradients for BC/IC loss
                bc_ic_grads = torch.autograd.grad(bc_ic_loss, self.model.parameters(), 
                                                retain_graph=True, create_graph=True, allow_unused=True)
                
                # Normalize gradients
                normalized_pde_grads = []
                for g in pde_grads:
                    if g is not None:
                        norm = torch.norm(g)
                        if norm > 1e-8:  # Avoid division by very small numbers
                            normalized_pde_grads.append(g / norm)
                        else:
                            normalized_pde_grads.append(g)
                    else:
                        normalized_pde_grads.append(None)
                
                normalized_bc_ic_grads = []
                for g in bc_ic_grads:
                    if g is not None:
                        norm = torch.norm(g)
                        if norm > 1e-8:  # Avoid division by very small numbers
                            normalized_bc_ic_grads.append(g / norm)
                        else:
                            normalized_bc_ic_grads.append(g)
                    else:
                        normalized_bc_ic_grads.append(None)
                
                # Combine normalized gradients
                final_grads = []
                for i in range(len(pde_grads)):
                    if pde_grads[i] is not None and bc_ic_grads[i] is not None:
                        final_grads.append(normalized_pde_grads[i] + normalized_bc_ic_grads[i])
                    elif pde_grads[i] is not None:
                        final_grads.append(normalized_pde_grads[i])
                    elif bc_ic_grads[i] is not None:
                        final_grads.append(normalized_bc_ic_grads[i])
                    else:
                        final_grads.append(None)
                
                # Apply the final gradients back to the parameters
                for param, grad in zip(self.model.parameters(), final_grads):
                    if grad is not None:
                        param.grad = grad
            
            except RuntimeError as e:
                # Fallback to standard backprop if gradient normalization fails
                print(f"Warning: Gradient normalization failed, using standard backprop. Error: {e}")
                optimizer.zero_grad()
                total_loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Record losses
            history['pde_loss'].append(pde_loss.item())
            history['bc_ic_loss'].append(bc_ic_loss.item())
            history['total_loss'].append(total_loss.item())
            
            # Log progress
            if epoch % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch}/{n_epochs}, "
                      f"PDE Loss: {pde_loss.item():.4e}, "
                      f"BC/IC Loss: {bc_ic_loss.item():.4e}, "
                      f"Total Loss: {total_loss.item():.4e}, "
                      f"Time: {elapsed_time:.2f}s")
                
        return history 


class AgenticPINN(PINN):
    """
    Rule-based adaptive PINN trainer.

    This controller adapts:
    - learning rate (based on loss plateau)
    - PDE/BC-IC loss weights (based on gradient-norm imbalance)
    """

    def __init__(
        self,
        model,
        pde,
        device=None,
        pde_weight=1.0,
        bc_ic_weight=1.0,
        grad_balance_momentum=0.9,
        min_weight=1e-2,
        max_weight=1e2,
        warmup_epochs=30,
        update_interval=5,
        max_adjust_ratio=1.25,
    ):
        super().__init__(model, pde, device)
        self.pde_weight = float(pde_weight)
        self.bc_ic_weight = float(bc_ic_weight)
        self.grad_balance_momentum = float(grad_balance_momentum)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.warmup_epochs = int(warmup_epochs)
        self.update_interval = int(update_interval)
        self.max_adjust_ratio = float(max_adjust_ratio)

        self._ema_grad_pde = None
        self._ema_grad_bc = None

    def _grad_norm(self, grads):
        norm_sq = torch.tensor(0.0, device=self.device)
        for g in grads:
            if g is not None:
                norm_sq = norm_sq + torch.sum(g.detach() ** 2)
        return torch.sqrt(norm_sq + 1e-12)

    def _update_loss_weights(self, grad_norm_pde, grad_norm_bc):
        # EMA smoothing for stability
        if self._ema_grad_pde is None:
            self._ema_grad_pde = grad_norm_pde
            self._ema_grad_bc = grad_norm_bc
        else:
            m = self.grad_balance_momentum
            self._ema_grad_pde = m * self._ema_grad_pde + (1.0 - m) * grad_norm_pde
            self._ema_grad_bc = m * self._ema_grad_bc + (1.0 - m) * grad_norm_bc

        # Target: balanced gradient magnitudes
        ratio = (self._ema_grad_pde + 1e-12) / (self._ema_grad_bc + 1e-12)
        ratio_sqrt = float(np.sqrt(ratio))

        # If PDE gradients dominate, increase BC/IC weight; vice versa.
        # Apply bounded step updates to avoid weight saturation.
        ratio_sqrt = float(np.clip(ratio_sqrt, 1.0 / self.max_adjust_ratio, self.max_adjust_ratio))

        self.pde_weight = float(np.clip(self.pde_weight / ratio_sqrt, self.min_weight, self.max_weight))
        self.bc_ic_weight = float(np.clip(self.bc_ic_weight * ratio_sqrt, self.min_weight, self.max_weight))

    def _update_learning_rate(self, optimizer, history, epoch, patience=100, decay=0.7, min_lr=1e-6):
        if epoch < 2 * patience:
            return

        recent = np.mean(history['total_loss'][-patience:])
        previous = np.mean(history['total_loss'][-2 * patience:-patience])
        improvement = (previous - recent) / (abs(previous) + 1e-12)

        # Plateau detector
        if improvement < 0.01:
            for group in optimizer.param_groups:
                new_lr = max(group['lr'] * decay, min_lr)
                group['lr'] = new_lr

    def train(self, optimizer, n_collocation_points=10000, n_epochs=10000, log_interval=100):
        history = {
            'pde_loss': [],
            'bc_ic_loss': [],
            'total_loss': [],
            'pde_weight': [],
            'bc_ic_weight': [],
            'lr': [],
            'grad_norm_pde': [],
            'grad_norm_bc': [],
        }

        for epoch in range(n_epochs):
            start_time = time.time()
            self.model.train()

            x = self.pde.generate_collocation_points(n_collocation_points)
            optimizer.zero_grad()

            pde_loss = self.pde_loss(x)
            bc_ic_loss = self.bc_ic_loss()

            # Gradient signals for adaptive weighting
            pde_grads = torch.autograd.grad(
                pde_loss,
                self.model.parameters(),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            bc_grads = torch.autograd.grad(
                bc_ic_loss,
                self.model.parameters(),
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )

            grad_norm_pde = self._grad_norm(pde_grads).item()
            grad_norm_bc = self._grad_norm(bc_grads).item()
            if epoch >= self.warmup_epochs and (epoch % self.update_interval == 0):
                self._update_loss_weights(grad_norm_pde, grad_norm_bc)

            total_loss = self.pde_weight * pde_loss + self.bc_ic_weight * bc_ic_loss
            total_loss.backward()
            optimizer.step()

            self._update_learning_rate(optimizer, history, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            history['pde_loss'].append(pde_loss.item())
            history['bc_ic_loss'].append(bc_ic_loss.item())
            history['total_loss'].append(total_loss.item())
            history['pde_weight'].append(self.pde_weight)
            history['bc_ic_weight'].append(self.bc_ic_weight)
            history['lr'].append(current_lr)
            history['grad_norm_pde'].append(grad_norm_pde)
            history['grad_norm_bc'].append(grad_norm_bc)

            if epoch % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch {epoch}/{n_epochs}, "
                    f"PDE Loss: {pde_loss.item():.4e}, "
                    f"BC/IC Loss: {bc_ic_loss.item():.4e}, "
                    f"Total Loss: {total_loss.item():.4e}, "
                    f"w_pde: {self.pde_weight:.3e}, "
                    f"w_bc: {self.bc_ic_weight:.3e}, "
                    f"lr: {current_lr:.2e}, "
                    f"Time: {elapsed_time:.2f}s"
                )

        return history