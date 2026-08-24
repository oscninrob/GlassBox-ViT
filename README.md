# GlassBox-ViT

Explainability (XAI) for Vision Transformers

---

## What is GlassBox-ViT?


GlassBox-ViT is an open-source explainability library designed specifically to address the interpretability challenges of Vision Transformers (ViTs). It provides a unified collection of XAI methods that work seamlessly with both standard architectures and custom models, enabling practitioners to understand what their vision models learn, where they attend, and why they make specific decisions.

It separates Black Box methods (compatible with any prediction function) from White Box techniques (leveraging native ViT representations), ensuring architectural clarity and respect for infrastructure constraints. Additionally, it provides complementary analysis tools for training sample influence tracing and model compression via knowledge distillation.

The fundamental problem it solves is straightforward: Vision Transformers remain opaque. Millions of parameters and complex attention mechanisms obscure the reasoning behind predictions. GlassBox-ViT bridges this gap by offering:

- **13 Production-Ready XAI Methods**: 6 White Box, 5 Black Box, and 2 complementary analysis tools.
- Unified and consistent visualizations for attention and attribution
- **Flexible architecture**: Black Box methods work with arbitrary prediction functions; White Box methods require native Hugging Face ViT models for deeper insights.
- **Infrastructure-aware design**: automatically adapts to user tensor devices and data types without explicit device management
- Comprehensive documentation and practical examples for each method

---

## Quick Start

### Installation

```bash
pip install glassbox-vit
```

Or from source:

```bash
git clone https://github.com/yourusername/glassbox-vit.git
cd glassbox-vit
pip install -e . 
```

### Basic Usage

All image explainers follow a consistent workflow: create an explainer instance, then call `generate()` to generate visual explanations.

```python
from glassbox_vit import GradCAMExplainer
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# Load model and processor
model = AutoModel.from_pretrained("google/vit-base-patch16-224")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Create explainer and generate explanation
explainer = GradCAMExplainer(model=model, processor=processor)
result = explainer.generate(image)

# Extract results
result['gradcam_image'].show()
print(f"Prediction: {result['predicted_label_id']}")
print(f"Confidence: {result['prediction_prob']:.4f}")
```

> **Note:** For detailed examples, refer to the `examples/` directory.

---

## Single-Image Explainers

These methods generate visual explanations for individual predictions.

### White Box Methods
White Box methods require native Hugging Face ViT models and processors for direct access to internal representations.

| Method | Return Keys | Key Feature |
| :--- | :--- | :--- |
| **GradCAM** | `gradcam_image`, `predicted_label_id`, `prediction_prob`, `explained_label_id` | Class-specific gradient-based saliency. |
| **ScoreCAM** | `scorecam_image`, `predicted_label_id`, `prediction_prob` | Score-weighted activation maps, gradient-free. |
| **Integrated Gradients** | `ig_image`, `predicted_label_id`, `prediction_prob` | Path-integral attribution from baseline to input. |
| **Expected Gradients** | `eg_image`, `predicted_label_id`, `prediction_prob` | Baseline-agnostic gradient attribution. |
| **Rollout** | `rollout_image`, `predicted_label_id`, `prediction_prob` | Attention flow analysis across ViT layers. |
| **SmoothGrad** | `smoothgrad_image`, `predicted_label_id`, `prediction_prob` | Gradient smoothing via input noise injection. |
| **PCA Explainer** | `pca_image`, `predicted_label_id`, `prediction_prob` | Principal component-based feature attribution. |

### Black Box Methods
Black Box methods accept any prediction function, making them compatible with arbitrary models and inference pipelines.

| Method | Return Keys | Key Feature |
| :--- | :--- | :--- |
| **LIME** | `lime_image`, `predicted_label_id`, `prediction_prob` | Local linear approximation via superpixel perturbation. |
| **SHAP** | `shap_image`, `predicted_label_id`, `prediction_prob` | Shapley value-based feature importance. |
| **RISE** | `rise_image`, `predicted_label_id`, `prediction_prob` | Randomized Input Sampling for Explanations. |
| **Patch Occlusion** | `patch_image`, `predicted_label_id`, `prediction_prob` | Evaluates feature importance by systematically occluding input patches. |

#### Using Black Box Methods with Hugging Face Models
Since Black Box methods are model-agnostic, they usually require a wrapper function that takes raw numpy images as input (often generated as perturbations by the explainer) and returns a numpy array of class probabilities. 

You can use the following bridge function to seamlessly connect your Hugging Face models and processors to any Black Box explainer:

```python
import torch
from PIL import Image

# --- BRIDGE FUNCTION ---
def hf_prediction_function(images_numpy):
    '''
    Converts numpy image perturbations into model probabilities.
    '''
    # Convert numpy arrays to PIL Images
    imgs_pil = [Image.fromarray((img).astype('uint8')) for img in images_numpy]
    
    # Process images and send to device
    inputs = processor(images=imgs_pil, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Return probabilities as a numpy array
    return torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
```

---

## Complementary Analysis Tools

These tools provide analysis beyond single-image explanation, with distinct workflows and outputs.

### TracIn (Training Sample Influence)
TracIn traces the influence of training samples on test predictions. Build an index from your training dataset, then query with test images to identify the most influential training samples.

**Typical Workflow:**
1. Initialize TracIn with model checkpoints.
2. Build an influence index from training data.
3. Query with test images to retrieve ranked influential samples.

**Output:** JSON containing training samples with influence scores per test image.

### Distilling (Knowledge Distillation)
Distilling creates a compressed student model from a larger teacher ViT, enabling efficient deployment while retaining predictive performance.

**Typical Workflow:**
1. Define teacher model and student architecture.
2. Train the student to mimic teacher outputs.
3. Deploy the lighter student model.

**Output:** Trained student model with knowledge transferred.

---

## Architecture

### Module Organization

```text
glassbox-vit/
├── glassbox_vit/
│   ├── __init__.py
│   ├── lime_explainer.py
│   ├── shap_explainer.py
│   ├── rise_explainer.py
│   ├── rollout_explainer.py
│   ├── patch_explainer.py
│   ├── gradcam_explainer.py
│   ├── scorecam_explainer.py
│   ├── smoothgrad_explainer.py
│   ├── expected_gradients.py
│   ├── integrated_gradients.py
│   ├── pca_explainer.py
│   ├── tracin_explainer.py
│   └── distilling.py
│   
├── examples/
│   
└── README.md
```

---

## Documentation

Comprehensive examples demonstrating each technique are located in the `examples/` directory. 

For theoretical background and citations, refer to the References section below.

<!-- ---

## References

* Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. https://arxiv.org/abs/2010.11929
* Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. https://arxiv.org/abs/1610.02055
* Ribeiro, M. T., et al. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. https://arxiv.org/abs/1602.04938
* Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. https://arxiv.org/abs/1705.07874
* Sundararajan, M., et al. (2017). Axiomatic Attribution for Deep Networks. https://arxiv.org/abs/1703.01365
* Pruthi, G., et al. (2020). Estimating Training Data Influence by Tracing Gradient Descent. https://arxiv.org/abs/2002.08484
* Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network. https://arxiv.org/abs/1503.02531

--- -->

## License

GlassBox-ViT is distributed under the MIT License. See `LICENSE` file for details.

## Contact

For bug reports, feature requests, or general inquiries, please use the GitHub Issues tracker.