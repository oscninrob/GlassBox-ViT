import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from skimage.transform import resize
from tqdm import tqdm

class RiseExplainer:
    """
    Explainer for Vision Models using RISE (Randomized Input Sampling for Explanation).
    This is a black-box explainer that evaluates thousands of randomly masked versions 
    of an image to build a saliency map based on the model's confidence.
    """
    
    def __init__(self, prediction_function, random_state=None):
        """
        Initializes the RISE explainer.
        
        Args:
            prediction_function (callable): A function that takes a list/array of NumPy images 
                                            and returns a NumPy array of probabilities.
            random_state (int, optional): Seed for reproducibility of the random masks.
        """
        self.prediction_function = prediction_function
        self.random_state = random_state
        
    def _generate_masks(self, num_masks, grid_size, p1, input_size):
        """
        Internal method to generate random blurred masks.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        cell_size = np.ceil(np.array(input_size) / grid_size)
        up_size = (grid_size + 1) * cell_size

        # Create binary grid
        grid = np.random.rand(num_masks, grid_size, grid_size) < p1
        grid = grid.astype('float32')

        masks = np.empty((num_masks, *input_size))
        for i in range(num_masks):
            # Resize to smooth the mask boundaries (bilinear interpolation)
            mask_up = resize(grid[i], up_size, order=1, mode='reflect', anti_aliasing=False)
            
            # Random crop to prevent grid alignment
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            masks[i, :, :] = mask_up[int(x):int(x + input_size[0]), int(y):int(y + input_size[1])]
            
        return masks

    def generate(self, pil_image, num_masks=2000, grid_size=8, p1=0.5, batch_size=32, show_progress=False):
        """
        Generates a RISE explanation for a single image.
        
        Args:
            pil_image (PIL.Image): The input image in PIL format.
            num_masks (int): Number of random masks to generate. Higher is better but slower.
            grid_size (int): Size of the grid for the mask generation (e.g., 8 means 8x8).
            p1 (float): Probability of a grid cell being unmasked (visible).
            batch_size (int): Number of masked images to send to the model at once.
            show_progress (bool): If True, displays a tqdm progress bar.
            
        Returns:
            dict: Containing the visual explanation (PIL.Image), predicted ID, and probability.
        """
        image_np = np.array(pil_image.convert("RGB"))
        input_size = image_np.shape[:2]

        # 1. Get the base prediction (unmasked image)
        real_probs = self.prediction_function(np.array([image_np]))[0]
        predicted_label_id = int(np.argmax(real_probs))

        # 2. Generate random masks
        masks = self._generate_masks(num_masks, grid_size, p1, input_size)

        # 3. Evaluate masked images
        salience_map = np.zeros(input_size)
        
        # Setup iterator (with or without tqdm)
        iterator = range(0, num_masks, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="RISE Evaluation")

        for b in iterator:
            batch_masks = masks[b:b+batch_size]
            batch_imgs = []

            # Apply masks to the image
            for mask in batch_masks:
                img_masked = (image_np * np.expand_dims(mask, axis=2)).astype(np.uint8)
                batch_imgs.append(img_masked)

            # Query the model using the injected prediction function
            batch_probs = self.prediction_function(batch_imgs)
            
            # Extract the scores for the winning class
            scores = batch_probs[:, predicted_label_id]
            
            # Accumulate the weighted masks
            for mask, score in zip(batch_masks, scores):
                salience_map += mask * score

        # --- MANUAL OVERLAY RENDERING (GRAYSCALE + JET MAP) ---
        # Normalize salience map between 0 and 1
        salience_map = salience_map - np.min(salience_map)
        vmax = np.max(salience_map)
        if vmax == 0:
            vmax = 1e-8
        salience_map = salience_map / vmax

        # Get Jet colormap (standard for RISE: Red is high importance, Blue is low)
        cmap = plt.get_cmap('jet')
        mapped_colors = cmap(salience_map)[:, :, :3] 

        # Create opacity mask based on salience (focus on the important parts)
        alpha = salience_map * 0.65  # Max opacity of 65% so the image underneath is visible
        alpha = np.expand_dims(alpha, axis=-1)

        # Convert original image to GRAYSCALE for maximum contrast
        img_float = image_np.astype(np.float32) / 255.0
        gray = np.dot(img_float[..., :3], [0.2989, 0.5870, 0.1140])
        gray_rgb = np.stack((gray,)*3, axis=-1)

        # Blend the heatmap over the Black & White photo
        blended = gray_rgb * (1 - alpha) + mapped_colors * alpha
        
        blended_uint8 = (blended * 255).clip(0, 255).astype(np.uint8)
        rise_image_pil = PILImage.fromarray(blended_uint8)

        return {
            'rise_image': rise_image_pil,
            'predicted_label_id': predicted_label_id,
            'prediction_prob': real_probs[predicted_label_id]
        }