import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage

class AttentionRolloutExplainer:
    """
    White-Box Explainer for Vision Transformers using Attention Rollout.
    Extracts and multiplies attention matrices across all layers to track 
    how information flows from the image patches to the final CLS token.
    """
    
    def __init__(self, model, processor):
        """
        Initializes the Attention Rollout explainer.
        """
        self.raw_model = model
        self.processor = processor
        self.device = next(self.raw_model.parameters()).device
        
    
        # UX LAYER: Attempt to force the model to output attention matrices
        try:
            self.raw_model.config.output_attentions = True
        except ValueError as e:
            # Handle PyTorch 2.0 SDPA (FlashAttention) compatibility issues cleanly
            if "sdpa" in str(e).lower():
                raise RuntimeError(
                    "[GlassBox-ViT] Your model uses SDPA (FlashAttention) which hides attention matrices. "
                    "Please reload your model using `attn_implementation='eager'`."
                ) from None
            raise e
        
        self.raw_model.eval()

    def _calculate_rollout(self, attentions):
        """
        Calculates the Attention Rollout by multiplying attention matrices across all layers.
        """
        # Start with an identity matrix
        result = torch.eye(attentions[0].size(-1), device=attentions[0].device)

        for attention in attentions:
            # Average the attention heads for this layer
            attention_heads_fused = attention.mean(dim=1)

            # Add residual connection (identity) and normalize
            flat = attention_heads_fused + torch.eye(attention_heads_fused.size(-1)).to(attention.device)
            flat = flat / flat.sum(dim=-1, keepdim=True)

            # Matrix multiplication (The actual Rollout)
            result = torch.matmul(flat, result)

        return result

    def generate(self, pil_image):
        """
        Generates the Attention Rollout map for a single image.
        
        Args:
            pil_image (PIL.Image): The input image in PIL format.
            
        Returns:
            dict: Containing the visual explanation (PIL.Image), predicted ID, and probability.
        """
        image_np = np.array(pil_image.convert("RGB"))
        original_height, original_width = image_np.shape[:2]

        # 1. Prepare image and pass it through the model
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.raw_model(**inputs)

        # Safety check: Verify the model actually returned attention matrices
        if not hasattr(outputs, 'attentions') or not outputs.attentions:
            raise ValueError(
                "[GlassBox-ViT] The model failed to return attention matrices. "
                "Ensure your model architecture natively supports attention extraction."
            )
        attentions = outputs.attentions


        # 2. Get predictions
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        predicted_label_id = int(torch.argmax(probs).item())
        prediction_prob = probs[predicted_label_id].item()

        # 3. Compute Attention Rollout
        rollout = self._calculate_rollout(outputs.attentions)

        # 4. Extract CLS token attention to the image patches
        # Index 0 is batch, 0 is CLS token, 1: are the image patches
        mask = rollout[0, 0, 1:].cpu().numpy()

        # Reshape the flat mask into a 2D grid (e.g., 14x14 or 16x16)
        grid_size = int(np.sqrt(mask.shape[0]))
        mask_grid = mask.reshape(grid_size, grid_size)

        # 5. Normalize and Resize the mask to match the original image
        mask_norm = mask_grid / np.max(mask_grid)
        mask_resized = cv2.resize(mask_norm, (original_width, original_height))



        # --- MANUAL OVERLAY RENDERING (GRAYSCALE + JET MAP) ---
        cmap = plt.get_cmap('jet')
        mapped_colors = cmap(mask_resized)[:, :, :3] 

        # Create opacity mask based on the attention intensity
        alpha = mask_resized * 0.7  # Max opacity of 70%
        alpha = np.expand_dims(alpha, axis=-1)

        # Convert original image to GRAYSCALE
        img_float = image_np.astype(np.float32) / 255.0
        gray = np.dot(img_float[..., :3], [0.2989, 0.5870, 0.1140])
        gray_rgb = np.stack((gray,)*3, axis=-1)

        # Blend the heatmap over the Black & White photo
        blended = gray_rgb * (1 - alpha) + mapped_colors * alpha
        
        blended_uint8 = (blended * 255).clip(0, 255).astype(np.uint8)
        rollout_image_pil = PILImage.fromarray(blended_uint8)

        return {
            'rollout_image': rollout_image_pil,
            'predicted_label_id': predicted_label_id,
            'prediction_prob': prediction_prob
        }