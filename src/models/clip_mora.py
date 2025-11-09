import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import src.models.utils as model_utils

from .modeling_clip import CLIPModel # We mainly modify the CLIPTextEmbeddings and CLIPVisionEmbeddings to support <CLS> from the other modality


class SharedMoRAParameters(nn.Module):
    """
    Shared parameters for MoRA that can be used by both text and vision modalities
    """
    def __init__(
        self,
        text_dim: int,      # Text dimension
        vision_dim: int,    # Vision dimension
        shared_rank: int = 16,  # Changed from r to shared_rank for clarity
    ):
        super().__init__()
        
        self.shared_rank = shared_rank
        
        # Initialize shared parameters with proper scaling
        std_dev = 1 / torch.sqrt(torch.tensor(shared_rank).float())
        self.shared_A = nn.Parameter(torch.randn(text_dim, shared_rank) * std_dev)    # [text_dim, shared_rank]
        self.shared_B = nn.Parameter(torch.randn(vision_dim, shared_rank) * std_dev)  # [vision_dim, shared_rank]


class TextMoRALayer(nn.Module):
    """
    MoRA implementation for text modality: combines shared parameters with modality-specific DoRA
    """
    def __init__(
        self,
        linear: nn.Linear,                    # Original text linear layer
        shared_params: SharedMoRAParameters,  # Shared MoRA parameters
        alpha: float = 8,                     # Scaling factor for shared params
        text_rank: int = 8,                   # Rank for text-specific adaptation (renamed from r)
        dropout: float = 0.0,                 # Dropout probability
    ):
        super().__init__()
        
        self.linear = linear
        self.shared_params = shared_params
        self.alpha = alpha
        
        # Initialize two separate magnitudes instead of one
        original_magnitude = self.linear.weight.norm(p=2, dim=0, keepdim=True)
        self.shared_magnitude = nn.Parameter(original_magnitude.clone())
        self.specific_magnitude = nn.Parameter(original_magnitude.clone())
        
        # Initialize modality-specific LoRA parameters with text_rank
        text_dim = linear.in_features
        std_dev = 1 / torch.sqrt(torch.tensor(text_rank).float())
        
        self.modality_A = nn.Parameter(torch.randn(text_dim, text_rank) * std_dev)
        self.modality_B = nn.Parameter(torch.zeros(text_rank, text_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.merged = False
    
    def compute_shared_lora(self):
        """Compute the cross-modal LoRA contribution: A·(B^T·B)·A^T"""
        # Compute Gram matrix of B
        gram_B = self.shared_params.shared_B.T @ self.shared_params.shared_B  # [r, r]
        # Apply to A with proper matrix multiplication
        return self.shared_params.shared_A @ gram_B @ self.shared_params.shared_A.T  # [text_dim, text_dim]
    
    def compute_specific_lora(self):
        """Compute the modality-specific LoRA contribution: A_text·B_text"""
        return self.modality_A @ self.modality_B  # [text_dim, text_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)

        if self.merged:
            return F.linear(x, self.linear.weight, self.linear.bias)
        
        # Compute LoRA terms
        shared_lora = self.compute_shared_lora()
        specific_lora = self.compute_specific_lora()
        
        # Compute directional component for shared adaptation (ONLY shared_lora)
        shared_numerator = self.linear.weight + self.alpha * shared_lora.T
        shared_denominator = shared_numerator.norm(p=2, dim=0, keepdim=True).clamp(min=1e-5)
        shared_directional = shared_numerator / shared_denominator
        
        # Compute directional component for specific adaptation (ONLY specific_lora)
        specific_numerator = self.linear.weight + self.alpha * specific_lora.T
        specific_denominator = specific_numerator.norm(p=2, dim=0, keepdim=True).clamp(min=1e-5)
        specific_directional = specific_numerator / specific_denominator
        
        # Blend the two directional components with separate magnitudes
        new_weight = self.shared_magnitude * shared_directional + self.specific_magnitude * specific_directional
        
        # Apply the new weight
        return F.linear(x, new_weight, self.linear.bias)
    
    def merge_weights(self):
        """Merge the weights for inference"""
        if self.merged:
            return
            
        # Compute directional component for shared adaptation (ONLY shared_lora)
        shared_lora = self.compute_shared_lora()
        shared_numerator = self.linear.weight + self.alpha * shared_lora.T
        shared_denominator = shared_numerator.norm(p=2, dim=0, keepdim=True).clamp(min=1e-5)
        shared_directional = shared_numerator / shared_denominator
        
        # Compute directional component for specific adaptation (ONLY specific_lora)
        specific_lora = self.compute_specific_lora()
        specific_numerator = self.linear.weight + self.alpha * specific_lora.T
        specific_denominator = specific_numerator.norm(p=2, dim=0, keepdim=True).clamp(min=1e-5)
        specific_directional = specific_numerator / specific_denominator
        
        # Blend with separate magnitudes
        self.linear.weight.data = self.shared_magnitude * shared_directional + self.specific_magnitude * specific_directional
        self.merged = True


class VisionMoRALayer(nn.Module):
    """
    MoRA implementation for vision modality: combines shared parameters with modality-specific DoRA
    """
    def __init__(
        self,
        linear: nn.Linear,                    # Original vision linear layer
        shared_params: SharedMoRAParameters,  # Shared MoRA parameters
        alpha: float = 8,                     # Scaling factor for shared params
        vision_rank: int = 8,                 # Rank for vision-specific adaptation (renamed from r)
        dropout: float = 0.0,                 # Dropout probability
    ):
        super().__init__()
        
        self.linear = linear
        self.shared_params = shared_params
        self.alpha = alpha
        
        # Initialize two separate magnitudes instead of one
        original_magnitude = self.linear.weight.norm(p=2, dim=0, keepdim=True)
        self.shared_magnitude = nn.Parameter(original_magnitude.clone())
        self.specific_magnitude = nn.Parameter(original_magnitude.clone())
        
        # Initialize modality-specific LoRA parameters with vision_rank
        vision_dim = linear.in_features
        std_dev = 1 / torch.sqrt(torch.tensor(vision_rank).float())
        
        self.modality_A = nn.Parameter(torch.randn(vision_dim, vision_rank) * std_dev)
        self.modality_B = nn.Parameter(torch.zeros(vision_rank, vision_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.merged = False
    
    def compute_shared_lora(self):
        """Compute the cross-modal LoRA contribution: B·(A^T·A)·B^T"""
        # Compute Gram matrix of A
        gram_A = self.shared_params.shared_A.T @ self.shared_params.shared_A  # [r, r]
        # Apply to B with proper matrix multiplication
        return self.shared_params.shared_B @ gram_A @ self.shared_params.shared_B.T  # [vision_dim, vision_dim]
    
    def compute_specific_lora(self):
        """Compute the modality-specific LoRA contribution: A_vision·B_vision"""
        return self.modality_A @ self.modality_B  # [vision_dim, vision_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for vision modality"""
        x = self.dropout(x)

        if self.merged:
            return F.linear(x, self.linear.weight, self.linear.bias)
        
        # Compute LoRA terms
        shared_lora = self.compute_shared_lora()
        specific_lora = self.compute_specific_lora()
        
        # Compute directional component for shared adaptation
        shared_numerator = self.linear.weight + self.alpha * shared_lora.T
        shared_denominator = shared_numerator.norm(p=2, dim=0, keepdim=True).clamp(min=1e-5)
        shared_directional = shared_numerator / shared_denominator
        
        # Compute directional component for specific adaptation
        specific_numerator = self.linear.weight + self.alpha * specific_lora.T
        specific_denominator = specific_numerator.norm(p=2, dim=0, keepdim=True).clamp(min=1e-5)
        specific_directional = specific_numerator / specific_denominator
        
        # Blend the two directional components with separate magnitudes
        new_weight = self.shared_magnitude * shared_directional + self.specific_magnitude * specific_directional
        
        # Apply the new weight
        return F.linear(x, new_weight, self.linear.bias)
    
    def merge_weights(self):
        """Merge the weights for inference"""
        if self.merged:
            return
            
        # Compute directional component for shared adaptation
        shared_lora = self.compute_shared_lora()
        shared_numerator = self.linear.weight + self.alpha * shared_lora.T
        shared_denominator = shared_numerator.norm(p=2, dim=0, keepdim=True).clamp(min=1e-5)
        shared_directional = shared_numerator / shared_denominator
        
        # Compute directional component for specific adaptation
        specific_lora = self.compute_specific_lora()
        specific_numerator = self.linear.weight + self.alpha * specific_lora.T
        specific_denominator = specific_numerator.norm(p=2, dim=0, keepdim=True).clamp(min=1e-5)
        specific_directional = specific_numerator / specific_denominator
        
        self.linear.weight.data = self.shared_magnitude * shared_directional + self.specific_magnitude * specific_directional
        self.merged = True


# Rename ClipMDoRA to ClipMoRA
class ClipMoRA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = CLIPModel.from_pretrained(cfg.BACKBONE_NAME)
        self.cls_head = nn.Linear(
            self.model.visual_projection.out_features+self.model.text_projection.out_features, 
            cfg.NUM_CLASSES,
        )
        self.cls_head.apply(model_utils.init_weights)

        # Create MoRA layers
        self.create_mora_layers()

        # Freeze backbone parameters
        self.learnable_modules = (
            "shared_A",
            "shared_B",
            "shared_magnitude",
            "specific_magnitude",
            "modality_A",
            "modality_B",
            "cls_head",
        )
        for name, param in self.named_parameters():
            for learnable_module in self.learnable_modules:
                if learnable_module in name:
                    param.requires_grad_(True)
                    break
                param.requires_grad_(False)
        
        # Double check
        enabled = list()
        for name, param in self.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        logger.info(f"Learnable Params: {enabled}")
    
    def get_module_by_name(self, name):
        """Helper method to get a module by its name"""
        components = name.split('.')
        module = self.model
        for component in components[1:]:  # Skip 'clip'
            module = getattr(module, component)
        return module
    
    def set_module_by_name(self, name, new_module):
        """Helper method to set a module by its name"""
        components = name.split('.')
        parent_module = self.model
        for component in components[1:-1]:
            parent_module = getattr(parent_module, component)
        setattr(parent_module, components[-1], new_module)
    
    def create_mora_layers(self):
        target_modules = self.cfg.lora.TARGET_MODULES
        
        # Get the number of layers in vision and text encoders
        num_vision_layers = len(self.model.vision_model.encoder.layers)
        num_text_layers = len(self.model.text_model.encoder.layers)
        
        # Group modules by layer index
        layer_modules = {}
        
        for module_template in target_modules:
            is_vision = "vision_model" in module_template
            num_layers = num_vision_layers if is_vision else num_text_layers
            
            for i in range(num_layers):
                if i < self.cfg.lora.START_LAYER_INDEX:
                    continue
                
                module_name = module_template.format(i=i)
                
                # Get the original module
                try:
                    module = self.get_module_by_name(module_name)
                except AttributeError:
                    continue
                
                # Ensure it's a Linear layer
                if not isinstance(module, nn.Linear):
                    continue
                
                # Initialize the layer_modules dict if needed
                if i not in layer_modules:
                    layer_modules[i] = {"vision": [], "text": []}
                
                # Store module info
                modality = "vision" if is_vision else "text"
                layer_modules[i][modality].append((module_name, module))
        
        # Create shared parameters and MoRA layers for each layer
        self.shared_params_dict = nn.ModuleDict()
        
        for layer_idx, modules in layer_modules.items():
            vision_modules = modules["vision"]
            text_modules = modules["text"]
            
            # Skip if either modality is missing at this layer
            if not vision_modules or not text_modules:
                continue
            
            # Get dimensions from first module of each modality
            vision_dim = vision_modules[0][1].in_features
            text_dim = text_modules[0][1].in_features
            
            # Create shared parameters with shared_rank
            shared_key = f"layer_{layer_idx}"
            self.shared_params_dict[shared_key] = SharedMoRAParameters(
                text_dim=text_dim,
                vision_dim=vision_dim,
                shared_rank=self.cfg.lora.SHARED_RANK,  # Use shared_rank parameter
            )
            
            # Create and apply MoRA layers
            for module_name, module in vision_modules:
                mora_layer = VisionMoRALayer(
                    module,
                    self.shared_params_dict[shared_key],
                    alpha=self.cfg.lora.ALPHA,
                    vision_rank=self.cfg.lora.VISION_RANK,  # Use vision_rank parameter
                    dropout=self.cfg.lora.DROPOUT,
                )
                self.set_module_by_name(module_name, mora_layer)
            
            for module_name, module in text_modules:
                mora_layer = TextMoRALayer(
                    module,
                    self.shared_params_dict[shared_key],
                    alpha=self.cfg.lora.ALPHA,
                    text_rank=self.cfg.lora.TEXT_RANK,  # Use text_rank parameter
                    dropout=self.cfg.lora.DROPOUT,
                )
                self.set_module_by_name(module_name, mora_layer)
    
    def forward(self, batch):
        inputs = batch["inputs"]
        inputs.pixel_values = inputs.pixel_values.half()

        # Vanilla CLIP forward
        image_features = self.model.get_image_features(
            inputs.pixel_values, 
            text_features=None,
        )
        text_features = self.model.get_text_features(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            image_features=None,
        )

        combined_features = torch.cat([image_features, text_features], dim=1)

        # Calculate the logits
        logits = self.cls_head(combined_features)
        return_dict = {"logits": logits}

        return return_dict
    
    def get_learned_weights(self):
        """
        Save only the trained parameters from the model.
        Also saves metadata about model configuration for validation during loading.
        
        Returns:
            dict: State dictionary containing only the learned parameters and configuration
        """
        # Get full state dict
        full_state_dict = self.state_dict()
        
        # Check if any MoRA layers have already been merged
        merged_layers = []
        for name, module in self.named_modules():
            if isinstance(module, (TextMoRALayer, VisionMoRALayer)) and module.merged:
                merged_layers.append(name)
        
        if merged_layers:
            logger.warning(f"Some MoRA layers have been merged: {merged_layers}")
            logger.warning("Saving merged weights may not work for continued training.")
        
        # Filter for trainable parameters (more reliable than name matching)
        trainable_params = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.add(name)
        
        # Create filtered state dict with trainable parameters
        adapter_state_dict = {k: v for k, v in full_state_dict.items() if k in trainable_params}
        
        # Also save essential configuration information
        adapter_state_dict['_metadata'] = {
            'backbone_name': self.cfg.BACKBONE_NAME,
            'num_classes': self.cfg.NUM_CLASSES,
            'shared_rank': self.cfg.lora.SHARED_RANK,
            'vision_rank': self.cfg.lora.VISION_RANK,
            'text_rank': self.cfg.lora.TEXT_RANK,
            'alpha': self.cfg.lora.ALPHA,
            'merged_layers': merged_layers
        }

        return adapter_state_dict

    def load_learned_weights(self, state_dict, merge_weights=True):
        """
        Load learned weights and optionally merge them for inference.
        
        Args:
            state_dict (dict): Dictionary containing model parameters
            merge_weights (bool): Whether to merge weights after loading
                                Set to False if you want to continue training

        Returns:
            int: Number of layers that had weights merged (if merge_weights=True)
                or number of parameters loaded (if merge_weights=False)
        """
        # Extract and verify metadata if present
        metadata = state_dict.pop('_metadata', {})
        if metadata:
            if metadata.get('backbone_name') != self.cfg.BACKBONE_NAME:
                logger.warning(f"Backbone mismatch. Saved: {metadata.get('backbone_name')}, "
                            f"Current: {self.cfg.BACKBONE_NAME}")
            
            if metadata.get('num_classes') != self.cfg.NUM_CLASSES:
                logger.warning(f"Number of classes mismatch. Saved: {metadata.get('num_classes')}, "
                            f"Current: {self.cfg.NUM_CLASSES}")
            
            # Log if we're loading from a checkpoint with merged layers
            if metadata.get('merged_layers'):
                logger.warning("Loading from a checkpoint with merged layers. "
                            "This may not work correctly for continued training.")
        
        # Track parameters for detailed reporting
        missing_keys = []
        unexpected_keys = []
        loaded_keys = []
        
        # Load parameters individually to handle shape mismatches
        for name, param in self.named_parameters():
            if param.requires_grad:  # Only consider trainable parameters
                if name in state_dict:
                    if param.shape == state_dict[name].shape:
                        param.data.copy_(state_dict[name])
                        loaded_keys.append(name)
                        state_dict.pop(name)  # Remove to track unexpected keys
                    else:
                        unexpected_keys.append(f"{name} (shape mismatch: got {state_dict[name].shape}, "
                                            f"expected {param.shape})")
                else:
                    missing_keys.append(name)
        
        # Any remaining keys in state_dict are unexpected
        unexpected_keys.extend(list(state_dict.keys()))
        
        # Log loading statistics
        if loaded_keys:
            logger.info(f"Successfully loaded {len(loaded_keys)} parameters")
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys or shape mismatches: {unexpected_keys}")
        
        # Merge weights for inference if requested
        if merge_weights:
            merged_count = 0
            for name, module in self.named_modules():
                if isinstance(module, (TextMoRALayer, VisionMoRALayer)) and not module.merged:
                    module.merge_weights()
                    merged_count += 1
            
            logger.info(f"MoRA weights merged into {merged_count} original linear layers")
            return merged_count
        
        return len(loaded_keys)