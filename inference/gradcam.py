import numpy as np
import torch
from matplotlib import cm
from PIL import Image


def get_target_layer(xray_model):
    return xray_model.features.denseblock4.denselayer32.conv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class_idx):
        self.model.zero_grad()
        logits = self.model(image_tensor)
        score = logits[0, target_class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay_heatmap(original_image, cam, alpha=0.4):
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(original_image.size, Image.BILINEAR)
    cam_norm = np.array(cam_img).astype(np.float32) / 255.0

    colored = (cm.jet(cam_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored)

    original_rgb = np.array(original_image.convert("RGB")).astype(np.float32)
    blended = (alpha * colored.astype(np.float32) + (1 - alpha) * original_rgb).astype(np.uint8)
    overlay_img = Image.fromarray(blended)

    return heatmap_img, overlay_img
