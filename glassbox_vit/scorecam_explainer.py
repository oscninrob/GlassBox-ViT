import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class _HuggingFaceModelWrapper(torch.nn.Module):
    """
    Private wrapper to bridge Hugging Face models and pytorch_grad_cam.
    Score-CAM expects the forward pass to return raw tensors (logits), 
    but HF models return a complex object.
    """
    def __init__(self, model):
        super(_HuggingFaceModelWrapper, self).__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits

def _reshape_transform(tensor):
    """
    Private function to reshape the 1D token sequence of a ViT back into a 2D spatial grid.
    Required by Score-CAM to understand the spatial layout of the Transformer.
    """
    # Tensor shape: [batch, num_tokens, dimensions]
    # Remove the CLS token (index 0)
    tokens = tensor[:, 1:, :]

    # Calculate grid size (e.g., 196 tokens = 14x14 grid, 256 tokens = 16x16 grid)
    grid_size = int(np.sqrt(tokens.shape[1]))

    # Reshape to [batch, height, width, channels]
    result = tokens.reshape(tokens.size(0), grid_size, grid_size, tokens.size(2))

    # CAM methods expect CNN format: [batch, channels, height, width]
    result = result.permute(0, 3, 1, 2)
    return result


class ScoreCamExplainer:
    """
    White-Box Explainer using Score-CAM (Score-Weighted Visual Explanations).
    It bypasses the 'Shattered Gradients' problem in Vision Transformers by using 
    forward passes to calculate feature importance instead of relying on backpropagation.
    """
    
    def __init__(self, model, processor, target_layers=None, batch_size=16):
        """
        Initializes the Score-CAM explainer.
        
        Args:
            model: The Hugging Face ViT/DINO/BEiT model.
            processor: The Hugging Face image processor.
            target_layers (list): Optional. The specific layer to compute activations from.
            batch_size (int): Since Score-CAM does multiple forward passes, this controls 
                              how many masked images are processed simultaneously. Reduce 
                              this if you encounter Out-Of-Memory (OOM) errors.
        """
        self.raw_model = model
        self.processor = processor
        self.device = next(self.raw_model.parameters()).device
        
        self.raw_model.eval()

        # Wrap the model for Score-CAM
        self.wrapped_model = _HuggingFaceModelWrapper(self.raw_model)

        # Auto-detect target layer if not provided
        if target_layers is None:
            self.target_layers = self._auto_detect_target_layer()
        else:
            self.target_layers = target_layers

        # Initialize the core Score-CAM object
        self.cam = ScoreCAM(
            model=self.wrapped_model,
            target_layers=self.target_layers,
            reshape_transform=_reshape_transform
        )
        
        # Apply batch size for the forward passes
        self.cam.batch_size = batch_size

    def _auto_detect_target_layer(self):
        """
        Attempts to dynamically find the final layer of the Transformer encoder.
        This avoids hardcoded paths that break across 'transformers' library versions.
        """
        target_module = None
        modules_dict = dict(self.raw_model.named_modules())
        
        # Search backwards to find the LAST layer norm before the attention block.
        for name, module in reversed(modules_dict.items()):
            if name.endswith('layernorm_before') or name.endswith('norm1'):
                target_module = module
                break
                
        if target_module is not None:
            return [target_module]
            
        raise ValueError(
            "Could not automatically detect the target layer for Score-CAM. "
            "Please pass 'target_layers' manually during initialization."
        )

    def generate(self, pil_image, target_class_id=None):
        """
        Generates a Score-CAM explanation for a single image.
        
        Args:
            pil_image (PIL.Image): The input image in PIL format.
            target_class_id (int): Optional. The class ID to explain. If None, 
                                   it explains the model's top prediction.
            
        Returns:
            dict: Containing the visual explanation (PIL.Image), predicted ID, and probability.
        """
        image_np = np.array(pil_image.convert("RGB"))

        # 1. Prepare image and pass it through the model
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        input_tensor = inputs['pixel_values']
        
        # 2. Get predictions (using the wrapper)
        with torch.no_grad():
            logits = self.wrapped_model(input_tensor)
        
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        model_predicted_id = int(torch.argmax(probs).item())
        prediction_prob = probs[model_predicted_id].item()

        # Decide which class to explain
        class_to_explain = target_class_id if target_class_id is not None else model_predicted_id

        # 3. Generate Score-CAM heatmap
        targets = [ClassifierOutputTarget(class_to_explain)]
        
        # Generates a numpy array [height, width] normalized between 0 and 1
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0, :]

        # --- MANUAL OVERLAY RENDERING (GRAYSCALE + JET MAP) ---
        cmap = plt.get_cmap('jet')
        mapped_colors = cmap(grayscale_cam)[:, :, :3] 

        # Opacity mask based on intensity
        alpha = grayscale_cam * 0.7 
        alpha = np.expand_dims(alpha, axis=-1)

        # Convert original image to GRAYSCALE
        img_float = image_np.astype(np.float32) / 255.0
        gray = np.dot(img_float[..., :3], [0.2989, 0.5870, 0.1140])
        gray_rgb = np.stack((gray,)*3, axis=-1)

        # Blend the heatmap over the Black & White photo
        blended = gray_rgb * (1 - alpha) + mapped_colors * alpha
        
        blended_uint8 = (blended * 255).clip(0, 255).astype(np.uint8)
        cam_image_pil = PILImage.fromarray(blended_uint8)

        return {
            'scorecam_image': cam_image_pil,
            'predicted_label_id': model_predicted_id,
            'prediction_prob': prediction_prob,
            'explained_label_id': class_to_explain
        }