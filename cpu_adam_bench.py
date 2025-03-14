import torch
from cpu_adam import create_adam, destroy_adam, adam_update

# Hyperparameters 
lr = 1e-3
bias_correction = True
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0
adamw_mode = True

class CPUAdam_Benchmark:
    def __init__(self, dtype: torch.dtype, param_size: int) -> None:
        """Initialize the benchmark with optimizer and tensors."""
        self.optimizer_id = 0
        self.dtype = dtype
        self.param_size = param_size
        create_adam(self.optimizer_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode, True)

        self.step_id = 0

        # Initialize tensors
        self.param = torch.zeros((param_size,), dtype=dtype, device="cpu")
        self.grad = torch.randn((param_size,), dtype=dtype, device="cpu")
        self.exp_avg = torch.zeros((param_size,), dtype=dtype, device="cpu")
        self.exp_avg_sq = torch.zeros((param_size,), dtype=dtype, device="cpu")

    def __del__(self):
        """Clean up the C++ optimizer object."""
        destroy_adam(self.optimizer_id)

    @torch.no_grad()
    def step(self):
        """Perform one optimization step using AdamW."""
        self.step_id += 1
        beta1, beta2 = betas
        adam_update(self.optimizer_id, self.step_id, lr, beta1, beta2, eps,
                    weight_decay, bias_correction, self.param, self.grad,
                    self.exp_avg, self.exp_avg_sq)