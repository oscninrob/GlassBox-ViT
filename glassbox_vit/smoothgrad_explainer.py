import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
from captum.attr import Saliency, NoiseTunnel
from captum.attr import visualization as viz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import gc

class SmoothGradExplainer:
    """
    Explainer for Hugging Face Vision Models using SmoothGrad.
    Averages the gradients of multiple noisy copies of the image to
    reduce artifacts and highlight robust features.
    """

    def __init__(self, model, processor):
        self.model = model
        self.model.eval()
        self.processor = processor
        self.device = next(self.model.parameters()).device

        def hf_forward_wrapper(pixel_values):
            outputs = self.model(pixel_values=pixel_values)
            return outputs.logits

        # SmoothGrad uses raw gradients (Saliency) inside a Noise Tunnel
        saliency = Saliency(hf_forward_wrapper)
        self.explainer = NoiseTunnel(saliency)

    def generate(self, pil_image, num_samples=50, stdevs=0.15):
        """
        Generates a SmoothGrad explanation using Captum's official safe rendering.
        """
        inputs = self.processor(images=pil_image, return_tensors="pt")
        input_tensor = inputs['pixel_values'].to(self.device)
        input_tensor.requires_grad = True

        with torch.no_grad():
            logits = self.model(pixel_values=input_tensor).logits
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            predicted_label_id = int(np.argmax(probs))
            prediction_prob = float(probs[predicted_label_id])

        # Core Computation
        attributions = self.explainer.attribute(
            input_tensor,
            nt_type='smoothgrad',
            nt_samples=num_samples,
            stdevs=stdevs,
            target=predicted_label_id
        )

        attributions_np = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        original_image_np = np.transpose(input_tensor.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        # Safe Rendering (Grayscale background, dynamic transparency)
        fig, axis = viz.visualize_image_attr(
            attributions_np,
            original_image_np,
            method="blended_heat_map",
            sign="absolute_value",
            show_colorbar=False,
            use_pyplot=False
        )

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        final_image = PILImage.open(buf).convert('RGB')

        # Memory Cleanup
        fig.clf()
        plt.close(fig)
        buf.close()
        gc.collect()

        final_image = final_image.resize(pil_image.size)

        return {
            'smoothgrad_image': final_image,
            'predicted_label_id': predicted_label_id,
            'prediction_prob': prediction_prob
        }