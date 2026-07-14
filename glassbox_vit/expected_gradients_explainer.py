import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from captum.attr import GradientShap
from captum.attr import visualization as viz

class ExpectedGradientsExplainer:
    """
    Explainer for Hugging Face Vision Models using Expected Gradients mathematics.
    Automatically generates blurred baselines and uses Captum's official safe
    rendering to handle outlier percentiles and color maps automatically.
    """

    def __init__(self, model, processor):
        """
        Initializes the explainer.

        Args:
            model (PreTrainedModel): The Hugging Face PyTorch model.
            processor (AutoImageProcessor): The corresponding HF image processor.
        """
        self.model = model
        self.model.eval()
        self.processor = processor
        self.device = next(self.model.parameters()).device

        def hf_forward_wrapper(pixel_values):
            outputs = self.model(pixel_values=pixel_values)
            return outputs.logits

        self.explainer = GradientShap(hf_forward_wrapper)

    def generate(self, pil_image, baselines=None, n_samples=50):
        """
        Generates the explanation using Captum's official UI tool in a memory-safe way.

        Args:
            pil_image (PIL.Image): The input image.
            baselines (torch.Tensor, optional): A batch of baseline images.
                                                If None, generates blurred versions automatically.
            n_samples (int): Number of random samples to draw from the baselines.

        Returns:
            dict: Contains 'eg_image' (PIL.Image), 'predicted_label_id', and 'prediction_prob'.
        """
        inputs = self.processor(images=pil_image, return_tensors="pt")
        input_tensor = inputs['pixel_values'].to(self.device)
        input_tensor.requires_grad = True

        with torch.no_grad():
            logits = self.model(pixel_values=input_tensor).logits
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            predicted_label_id = int(np.argmax(probs))
            prediction_prob = float(probs[predicted_label_id])

        # --- AUTOMATED BASELINES GENERATION ---
        if baselines is None:
            baseline_list = []

            for kernel_size in [15, 31, 61]:
                blurred = TF.gaussian_blur(input_tensor, kernel_size=(kernel_size, kernel_size))
                baseline_list.append(blurred)

            baseline_list.append(torch.zeros_like(input_tensor))
            baselines = torch.cat(baseline_list, dim=0)
        else:
            baselines = baselines.to(self.device)

        # 4. Core Computation
        attributions = self.explainer.attribute(
            input_tensor,
            baselines=baselines,
            target=predicted_label_id,
            n_samples=n_samples
        )

        # 5. Format to NumPy for Captum Visualization
        attributions_np = np.transpose(attributions.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
        original_image_np = np.transpose(input_tensor.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        # --- THE SAFE LIBRARY PATH (OFFICIAL CAPTUM RENDERING + STRICT MEMORY MANAGEMENT) ---
        import matplotlib
        matplotlib.use('Agg')  # Force background mode to prevent server UI crashes
        import matplotlib.pyplot as plt
        import io
        import gc

        # Let Captum handle the percentiles, outliers, and color blending automatically
        fig, axis = viz.visualize_image_attr(
            attributions_np,
            original_image_np,
            method="blended_heat_map",
            sign="absolute_value",
            show_colorbar=False,
            use_pyplot=False # Prevent Captum from drawing to screen
        )

        # Extract the image safely
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        final_image = PILImage.open(buf).convert('RGB')

        # EXTREME MEMORY CLEANUP
        fig.clf()
        plt.close(fig)
        buf.close()
        gc.collect()

        # Resize to match the original user input exactly
        final_image = final_image.resize(pil_image.size)

        return {
            'eg_image': final_image,
            'predicted_label_id': predicted_label_id,
            'prediction_prob': prediction_prob
        }