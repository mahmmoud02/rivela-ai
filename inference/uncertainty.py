import torch
import torch.nn as nn


def enable_dropout(model):
    """Leaves BatchNorm/etc. in eval mode (using running stats) but switches
    Dropout layers back to train mode, so each forward pass randomly drops
    different neurons. Running the model N times this way and looking at how
    much the output swings around is MC Dropout uncertainty estimation."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_uncertainty(model, input_tensor, batch_forward_fn, disease_classes, n_samples=25):
    """Run all N MC Dropout passes in a single batched forward call instead of
    N sequential calls. input_tensor shape: [1, ...]. batch_forward_fn receives
    a tensor of shape [n_samples, ...] and returns a numpy array [n_samples, n_diseases]."""
    model.eval()
    enable_dropout(model)
    batched = input_tensor.repeat([n_samples] + [1] * (input_tensor.dim() - 1))
    samples = batch_forward_fn(batched)   # [n_samples, n_diseases]
    model.eval()

    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    return {
        disease: {"mean": float(mean[i]), "std": float(std[i])}
        for i, disease in enumerate(disease_classes)
    }
