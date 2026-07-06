import torch
import cv2
import numpy as np
from PIL import Image as PILImage
from sklearn.decomposition import PCA

class PCAExplainer:
    """
    White-Box Explainer using Principal Component Analysis (PCA) on hidden states.
    Extracts the deepest features of the Transformer and reduces them to 3 dimensions (RGB)
    to visualize semantic clustering (similar colors mean the model groups those patches together).
    """
    
    def __init__(self, model, processor):
        """
        Initializes the PCA explainer.
        
        Args:
            model: Hugging Face ViT model loaded with `output_hidden_states=True`.
            processor: Hugging Face image processor.
        """
        self.raw_model = model
        self.processor = processor
        
        # Read the device dynamically from the user's model
        self.device = next(self.raw_model.parameters()).device
        self.raw_model.eval()

    def generate(self, pil_image, interpolation=cv2.INTER_NEAREST):
        """
        Generates the PCA feature visualization for a single image.
        
        Args:
            pil_image (PIL.Image): The input image in PIL format.
            interpolation (int): cv2 interpolation method. Defaults to INTER_NEAREST 
                                 to clearly show the ViT patches. Use cv2.INTER_CUBIC 
                                 if you prefer smooth, cloud-like transitions.
            
        Returns:
            dict: Containing the visual explanation (PIL.Image), predicted ID, and probability.
        """
        image_np = np.array(pil_image.convert("RGB"))
        original_height, original_width = image_np.shape[:2]

        # 1. Prepare inputs and send to the model's device
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        # 2. Forward pass
        with torch.no_grad():
            outputs = self.raw_model(**inputs)

        # Safety check: Ensure hidden states were requested
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            raise ValueError("The model did not return hidden states. "
                             "Please load it with `output_hidden_states=True`.")

        # 3. Get predictions
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        predicted_label_id = int(torch.argmax(probs).item())
        prediction_prob = probs[predicted_label_id].item()

        # 4. Extract last hidden state
        # Shape: [batch_size, sequence_length, hidden_size] (e.g., [1, 197, 768])
        last_hidden_state = outputs.hidden_states[-1][0]

        # Remove CLS token (index 0) to keep only the spatial image patches
        patch_tokens = last_hidden_state[1:].cpu().numpy()

        # 5. Apply PCA to reduce dimensions to 3 (mapping to R, G, B channels)
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(patch_tokens)

        # Normalize to [0, 1] range for valid RGB visualization
        pca_min = pca_features.min(axis=0)
        pca_max = pca_features.max(axis=0)
        # Avoid division by zero in edge cases
        pca_features = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)

        # 6. Reshape to 2D spatial grid (e.g., 14x14 or 16x16)
        grid_size = int(np.sqrt(pca_features.shape[0]))
        pca_image = pca_features.reshape(grid_size, grid_size, 3)

        # 7. Resize to match the original image resolution
        pca_image_resized = cv2.resize(
            pca_image, 
            (original_width, original_height), 
            interpolation=interpolation
        )

        # Convert to PIL Image
        pca_image_uint8 = (pca_image_resized * 255).astype(np.uint8)
        pca_image_pil = PILImage.fromarray(pca_image_uint8)

        return {
            'pca_image': pca_image_pil,
            'predicted_label_id': predicted_label_id,
            'prediction_prob': prediction_prob
        }