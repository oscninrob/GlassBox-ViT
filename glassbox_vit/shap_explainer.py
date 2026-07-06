import numpy as np
import matplotlib.pyplot as plt
import shap
from PIL import Image as PILImage

class ShapExplainer:
    """
    Explainer for Vision Models using SHAP (SHapley Additive exPlanations).
    Returns a single image with the SHAP overlay (Red = positive impact, Blue = negative).
    """
    
    def __init__(self, prediction_function, class_names, image_shape=(224, 224, 3)):
        self.prediction_function = prediction_function
        self.masker = shap.maskers.Image("blur(11,11)", image_shape)
        
        self.explainer = shap.Explainer(
            self.prediction_function, 
            self.masker, 
            output_names=class_names
        )

    def generate(self, pil_image, max_evals=300, batch_size=16):
        
        image_np = np.array(pil_image.convert("RGB"))
        real_probs = self.prediction_function(np.array([image_np]))[0]
        predicted_label_id = int(np.argmax(real_probs))

        # --- THE SHAP MAGIC ---
        shap_values = self.explainer(
            np.array([image_np]),
            max_evals=max_evals,
            batch_size=batch_size
        )

        # --- MANUAL OVERLAY RENDERING ---
        # 1. Extract SHAP values for the predicted class
        sv = shap_values.values[0, ..., predicted_label_id]

        # 2. Sum across RGB channels to get a single 2D heatmap
        if sv.ndim == 3:
            heatmap = sv.sum(axis=-1)
        else:
            heatmap = sv

        # 3. Normalize the heatmap symmetrically around zero
        vmax = np.max(np.abs(heatmap))
        if vmax == 0:
            vmax = 1e-8 
        heatmap_norm = heatmap / vmax 

        # 4. Get Red-Blue colormap (Blue = negative, Red = positive)
        cmap = plt.get_cmap('bwr')
        mapped_colors = cmap((heatmap_norm + 1) / 2)[:, :, :3] 

        # 5. Increase max opacity to 85% and apply power curve for punchier mid-tones
        alpha = (np.abs(heatmap_norm) ** 0.8) * 0.85 
        alpha = np.expand_dims(alpha, axis=-1)

        # 6. Convert original image to GRAYSCALE (Standard XAI trick for contrast)
        img_float = image_np.astype(np.float32) / 255.0
        gray = np.dot(img_float[..., :3], [0.2989, 0.5870, 0.1140])
        gray_rgb = np.stack((gray,)*3, axis=-1) # Back to 3 channels for blending

        # 7. Blend the heatmap over the Black & White photo
        blended = gray_rgb * (1 - alpha) + mapped_colors * alpha
        
        blended_uint8 = (blended * 255).clip(0, 255).astype(np.uint8)
        shap_image_pil = PILImage.fromarray(blended_uint8)

        return {
            'shap_image': shap_image_pil,
            'predicted_label_id': predicted_label_id,
            'prediction_prob': real_probs[predicted_label_id]
        }
        