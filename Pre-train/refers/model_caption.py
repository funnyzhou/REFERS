import torch
import torch.nn as nn

from refers.modules import textual_heads
from refers.modules import VisionTransformer
import refers.models as vmodels

class Model_caption(nn.Module):
    def __init__(self, config):  # 256 * 7 * 7 = 12544
        super(Model_caption, self).__init__()
        visual_crop_size = config.DATA.IMAGE_CROP_SIZE
        visual_feature_size = config.MODEL.VISUAL.FEATURE_SIZE
        visual() = VisionTransformer(visual_crop_size, visual_feature_size, zero_head=True)
        
        textual_name = config.MODEL.TEXTUAL.NAME
        vocab_size = config.DATA.VOCAB_SIZE
        if "transformer" in config.MODEL.TEXTUAL.NAME:
            # Get architectural hyper-params as per name by matching regex.
            textual_name, architecture = textual_name.split("::")
            architecture = re.match(r"L(\d+)_H(\d+)_A(\d+)_F(\d+)", architecture)

            num_layers = int(architecture.group(1))
            hidden_size = int(architecture.group(2))
            attention_heads = int(architecture.group(3))
            feedforward_size = int(architecture.group(4))
        textual() = textual_heads(visual_feature_size = visual_feature_size,
                                vocab_size = vocab_size,
                                hidden_size = hidden_size,
                                num_layers = num_layers,
                                attention_heads = attention_heads,
                                feedforward_size = feedforward_size,
                                dropout = config.MODEL.TEXTUAL.DROPOUT,
                                mask_future_positions="captioning" in _C.MODEL.NAME,
                                max_caption_length=_C.DATA.MAX_CAPTION_LENGTH,
                                padding_idx=_C.DATA.UNK_INDEX,)
        Model() = vmodels.BidirectionalCaptioningModel(visual = visual,
                                            textual = textual,
                                            beam_size = 5,
                                            max_decoding_steps = 30,
                                            sos_index = 1,
                                            eos_index = 2,)

    def forward(self, x):
        output_dict = Model(x)
        return output_logits