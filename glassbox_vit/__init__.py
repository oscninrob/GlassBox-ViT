"""
GlassBox-ViT: A comprehensive XAI library for Vision Transformers.
Includes both White-Box (gradients/attention) and Black-Box (sampling) methods.
"""

# White-Box Explainer Methods
from .rollout_explainer import RolloutExplainer
from .gradcam_explainer import GradCAMExplainer
from .pca_explainer import PCAExplainer
from .smoothgrad_explainer import SmoothGradExplainer
from .expected_gradients_explainer import ExpectedGradientsExplainer
from .integrated_gradients_explainer import IntegratedGradientsExplainer
from .scorecam_explainer import ScoreCamExplainer

# Black-Box Explainer Methods

from .lime_explainer import LimeExplainer
from .shap_explainer import ShapExplainer
from .rise_explainer import RiseExplainer
from .patch_explainer import PatchOcclusionExplainer

# Other methods

from .tracIn_explainer import TracInExplainer
from .destilling import KnowledgeDistillationTrainer

__all__ = [
    "RolloutExplainer",
    "GradCAMExplainer",
    "PCAExplainer",
    "SmoothGradExplainer",
    "ExpectedGradientsExplainer",
    "IntegratedGradientsExplainer",
    "ScoreCamExplainer",
    "LimeExplainer",
    "ShapExplainer",
    "RiseExplainer",
    "PatchOcclusionExplainer",
    "TracInExplainer",
    "KnowledgeDistillationTrainer"
]