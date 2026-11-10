import sys
sys.path.insert(0, r'C:\Users\Raza-SEAGUARD\Downloads\Attention\attention-scaling-verify\src')

import torch
import numpy as np
from data import get_config, DataCache, prepare_dataset
from model import TinyTransformerClassifier

config = get_config('sst2')
device = torch.device("cuda")
cache = DataCache(config["data"]["cache_dir"])
prepared = prepare_dataset('sst2', config, cache)

# Create model with grad probe
model = TinyTransformerClassifier(
    vocab_size=prepared.vocab_size,
    num_classes=config["dataset"]["num_classes"],
    d_model=config["model"]["d_model"],
    n_heads=config["model"]["n_heads"],
    n_layers=config["model"]["n_layers"],
    dim_ff=config["model"]["dim_ff"],
    max_len=config["model"]["max_len"],
    dropout=config["model"]["dropout"],
    variant="standard",
    temperature=config.get("temperature", 2.0),
    with_grad_probe=True,
).to(device)

model.train()

# Enable grad probe
for module in model.modules():
    if hasattr(module, 'set_retain_grads'):
        module.set_retain_grads(True)

# Create probe batch
probe_X = torch.from_numpy(prepared.X_train[:16]).to(device)
probe_y = torch.from_numpy(prepared.y_train[:16]).to(device)

# Forward pass
logits = model(probe_X)
loss = torch.nn.functional.cross_entropy(logits, probe_y)
loss.backward()

# Check gradients
for name, module in model.named_modules():
    if hasattr(module, '_grad_probes'):
        print(f"Grad probes for {name}:")
        for k, v in module._grad_probes.items():
            if v is not None:
                print(f"  {k}: shape={v.shape}, mean={v.abs().mean().item():.6f}")
            else:
                print(f"  {k}: None")

print("Debug successful!")