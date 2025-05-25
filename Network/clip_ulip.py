"""ULIP-2 text and 3D shape encoders for text-to-3D generation"""
import torch
import torch.nn as nn
import clip

class ULIP2Wrapper(nn.Module):
    def __init__(self, preloaded_clip_model: nn.Module):
        super().__init__()
        # preloaded_clip_model is assumed to be already loaded 
        # and moved to the correct device in the calling script.
        self.model = preloaded_clip_model

    def encode_text(self, tokenized_text_input):
        # This method expects 'tokenized_text_input' to be the output of clip.tokenize()
        # and already on the same device as self.model.
        return self.model.encode_text(tokenized_text_input)

    def encode_3d(self, point_clouds):
        # This method is not directly used by MaskGIT3D's text conditioning path.
        # If it were needed for a multimodal model, it would require a proper implementation.
        raise NotImplementedError("3D encoding is not implemented in this CLIP wrapper.")

    def forward(self, text=None, point_clouds=None):
        # In the context of MaskGIT3D calling this as a text_encoder,
        # 'text' will be pre-tokenized text (e.g., output of clip.tokenize()).
        
        if text is not None and point_clouds is not None:
            # This scenario is not used by MaskGIT3D's current text_encoder call pattern.
            # If it were, it might look like:
            # text_features = self.encode_text(text)
            # pc_features = self.encode_3d(point_clouds)
            # return text_features, pc_features
            raise NotImplementedError("Simultaneous text and 3D processing is not supported/used here.")
        elif text is not None:
            return self.encode_text(text)
        elif point_clouds is not None:
            return self.encode_3d(point_clouds)
        else:
            # It's good practice to handle the case where no valid input is provided.
            raise ValueError("ULIP2Wrapper.forward() called without 'text' or 'point_clouds'.")
