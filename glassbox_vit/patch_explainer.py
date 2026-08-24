import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from tqdm import tqdm
import warnings

class PatchOcclusionExplainer:
    """
    Black-Box Explainer using Patch-based Occlusion.
    Specifically suited for Vision Transformers (ViTs), as it systematically 
    blanks out grid patches (e.g., 16x16) and measures the drop in confidence.
    """
    
    def __init__(self, prediction_function):
        """
        Initializes the Patch Occlusion explainer.
        
        Args:
            prediction_function (callable): A function returning a NumPy array of probabilities.
        """
        self.prediction_function = prediction_function

    def generate(self, pil_image, patch_size=16, mask_value=0, batch_size=32, show_progress=False , resize_to_original=True ):
        """
        Generates a Patch Occlusion explanation for a single image.
        
        Args:
            pil_image (PIL.Image): The input image in PIL format.
            patch_size (int): Size of the square patches to occlude (usually 16 or 14 for ViTs).
            mask_value (int): Pixel value to fill the occluded patch (0 = black, 128 = gray).
            batch_size (int): Number of occluded images to evaluate at once.
            show_progress (bool): If True, displays a tqdm progress bar.
            
        Returns:
            dict: Containing the visual explanation (PIL.Image), predicted ID, and probability.
        """

        if not resize_to_original:
            warnings.warn(
                "PatchOcclusion is a Black-Box explainer that naturally operates on the provided image dimensions. "
                "The 'resize_to_original=False' flag is ignored to maintain API consistency."
            )


        image_np = np.array(pil_image.convert("RGB"))
        height, width, _ = image_np.shape

        # Get base prediction (intact image)
        real_probs = self.prediction_function(np.array([image_np]))[0]
        predicted_label_id = int(np.argmax(real_probs))
        base_prob = real_probs[predicted_label_id]

        # Calculate grid dimensions
        grid_h = height // patch_size
        grid_w = width // patch_size
        total_patches = grid_h * grid_w

        # Pre-calculate coordinates to avoid nested loops during inference
        patch_coords = [(r, c) for r in range(grid_h) for c in range(grid_w)]


        # Evaluate all occluded images in batches
        occ_probs = []
        iterator = range(0, total_patches, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Patch Occlusion")

        for b in iterator:
            batch_coords = patch_coords[b:b + batch_size]
            batch_imgs = []
            # Generate only the images needed for the current batch
            for (row, col) in batch_coords:
                img_occ = image_np.copy()
                y_start = row * patch_size
                y_end = (row + 1) * patch_size
                x_start = col * patch_size
                x_end = (col + 1) * patch_size
                
                img_occ[y_start:y_end, x_start:x_end, :] = mask_value
                batch_imgs.append(img_occ)

            # Inference
            batch_preds = self.prediction_function(np.array(batch_imgs))
            batch_scores = batch_preds[:, predicted_label_id]
            occ_probs.extend(batch_scores)

            
        # Calculate importance (Drop in probability)
        heatmap = np.zeros(total_patches)
        for i in range(total_patches):
            # Importance = Base Prob - Occluded Prob
            # We use max(0, x) to ignore patches that actually *improved* the prediction when removed
            heatmap[i] = max(0, base_prob - occ_probs[i])

        heatmap = heatmap.reshape((grid_h, grid_w))

        # Normalize and resize the heatmap
        vmax = np.max(heatmap)
        if vmax > 0:
            heatmap = heatmap / vmax

        # We use INTER_NEAREST to keep the sharp, blocky edges of the patches
        heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)

        # --- MANUAL OVERLAY RENDERING (GRAYSCALE + JET MAP) ---
        cmap = plt.get_cmap('jet')
        mapped_colors = cmap(heatmap_resized)[:, :, :3] 

        # Opacity mask based on importance
        alpha = heatmap_resized * 0.7  # Max opacity 70%
        alpha = np.expand_dims(alpha, axis=-1)

        # Convert original image to GRAYSCALE
        img_float = image_np.astype(np.float32) / 255.0
        gray = np.dot(img_float[..., :3], [0.2989, 0.5870, 0.1140])
        gray_rgb = np.stack((gray,)*3, axis=-1)

        # Blend
        blended = gray_rgb * (1 - alpha) + mapped_colors * alpha
        blended_uint8 = (blended * 255).clip(0, 255).astype(np.uint8)
        patch_image_pil = PILImage.fromarray(blended_uint8)

        return {
            'patch_image': patch_image_pil,
            'predicted_label_id': predicted_label_id,
            'prediction_prob': base_prob
        }