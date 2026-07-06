import numpy as np
from PIL import Image as PILImage
from lime import lime_image
from skimage.segmentation import mark_boundaries

class LimeExplainer:
    """
    Explainer for Vision Models using Local Interpretable Model-agnostic Explanations (LIME).
    This is a black-box explainer, meaning it relies entirely on a prediction function 
    and does not require direct access to the model's internal architecture.
    """
    
    def __init__(self, prediction_function, random_state=None):
        """
        Initializes the LIME explainer.
        
        Args:
            prediction_function (callable): A function that takes a NumPy array of images 
                                            (batch_size, height, width, channels) and 
                                            returns a NumPy array of probabilities.
            random_state (int): Seed for reproducibility across executions.
        """
        self.prediction_function = prediction_function
        self.explainer = lime_image.LimeImageExplainer(random_state=random_state)
        
    def generate(self, pil_image, num_samples=500):
        """
        Generates a LIME explanation for a single image.
        
        Args:
            pil_image (PIL.Image): The input image in PIL format.
            num_samples (int): Number of perturbations LIME will generate. Higher values
                               yield more stable explanations but take longer to compute.
            
        Returns:
            dict: A dictionary containing:
                  - 'lime_image' (PIL.Image): The visual explanation showing key superpixels.
                  - 'predicted_label_id' (int): The class ID predicted by the model.
                  - 'prediction_prob' (float): The probability of the predicted class.
        """
        # Convert PIL image to NumPy array (RGB) standard format for LIME
        image_np = np.array(pil_image.convert("RGB"))

        # Generate the explanation using the injected prediction function
        explanation = self.explainer.explain_instance(
            image_np, 
            self.prediction_function, 
            top_labels=1, 
            hide_color=0, 
            num_samples=num_samples
        )

        # Extract the top predicted label index
        predicted_label_id = explanation.top_labels[0]

        # Get image and mask for the top prediction (showing only positive contributions)
        temp, mask = explanation.get_image_and_mask(
            predicted_label_id, 
            positive_only=True, 
            num_features=5, 
            hide_rest=True
        )
        
        # Apply boundaries and convert back to standard 8-bit image format (0-255)
        lime_img_float = mark_boundaries(temp / 255.0, mask)
        lime_img_final = (lime_img_float * 255).astype(np.uint8)
        
        # Calculate the actual probability for the original unperturbed image
        real_probs = self.prediction_function(np.array([image_np]))[0]

        return {
            'lime_image': PILImage.fromarray(lime_img_final),
            'predicted_label_id': predicted_label_id,
            'prediction_prob': real_probs[predicted_label_id]
        }