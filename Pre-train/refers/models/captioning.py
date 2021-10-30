import copy
import functools
from typing import Any, Dict
import pdb

import torch
from torch import nn
from torch.nn import functional as F

from refers.data.structures import ImageCaptionBatch
from refers.data.tokenizers import SentencePieceBPETokenizer
from refers.modules.textual_heads import TextualHead
from refers.modules.visual_backbones import VisualBackbone
from refers.modules.Fusion import fusion
from refers.modules.contrastive_learning import TextEncoder
from refers.utils.beam_search import AutoRegressiveBeamSearch
from refers.utils.losses import LossesOfConVIRT

class CaptioningModel(nn.Module):
    r"""
    A model to perform image captioning (in both forward and backward directions
    independently, only in forward direction). It is composed of a
    :class:`~refers.modules.visual_backbones.VisualBackbone` and a
    :class:`~refers.modules.textual_heads.TextualHead` on top of it.

    During training, it maximizes the likelihood of ground truth caption
    conditioned on image features. During inference, it predicts a caption for
    an input image through beam search decoding.

    Parameters
    ----------
    visual: refers.modules.visual_backbones.VisualBackbone
        A :class:`~refers.modules.visual_backbones.VisualBackbone` which
        computes visual features from an input image.
    textual: refers.modules.textual_heads.TextualHead
        A :class:`~refers.modules.textual_heads.TextualHead` which
        makes final predictions conditioned on visual features.
    beam_size : int, optional (default = 5)
        The width of the beam used for beam search.
    max_decoding_steps: int, optional (default = 30)
        The maximum number of decoding steps for beam search.
    sos_index: int, optional (default = 1)
        The index of the end token (``[SOS]``) in vocabulary.
    eos_index: int, optional (default = 2)
        The index of the end token (``[EOS]``) in vocabulary.
    caption_backward: bool, optional (default = False)
        Whether to *also* perform captioning in backward direction. Default is
        ``False`` -- only forward captioning is performed. When ``True``, a
        clone of textual head is created, which does not share weights with
        "forward" model except input and output embeddings.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        fusion: fusion,
        contrastive_bert: TextEncoder,
        textual: TextualHead,
        beam_size: int = 5,
        max_decoding_steps: int = 30,
        sos_index: int = 1,
        eos_index: int = 2,
        caption_backward: bool = False,
    ):
        super().__init__()
        # pdb.set_trace()
        self.visual = visual
        self.fusion = fusion
        self.contrastive_bert = contrastive_bert
        self.textual = textual
        self.padding_idx = self.textual.padding_idx
        self.caption_backward = caption_backward

        # Clone the textual module for backward direction if doing captioning
        # in both directions (separately).
        if self.caption_backward:
            self.backward_textual = copy.deepcopy(self.textual)

            # Share weights for visual projection, and input/output embeddings.
            self.backward_textual.visual_projection = self.textual.visual_projection
            self.backward_textual.embedding = self.textual.embedding
            self.backward_textual.output = self.textual.output

        # These boundary indices are needed for beam search.
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.beam_search = AutoRegressiveBeamSearch(
            self.eos_index, beam_size=5, max_steps=max_decoding_steps
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self.infoNCE = LossesOfConVIRT()

    def forward(self, batch: ImageCaptionBatch) -> Dict[str, Any]:
        r"""
        Given a batch of images and captions, compute log likelihood loss per
        caption token during training. During inference, given a batch of
        images, decode the most likely caption in forward direction through
        beam search decoding.

        Parameters
        ----------
        batch: refers.data.structures.ImageCaptionBatch
            A batch of images and (optionally) ground truth caption tokens.

        Returns
        -------
        Dict[str, Any]

            A dict with the following structure, containing loss for optimization,
            loss components to log directly to tensorboard, and optionally
            predictions.

            .. code-block::

                {
                    "loss": torch.Tensor,
                    "loss_components": {
                        "captioning_forward": torch.Tensor,
                        "captioning_backward": torch.Tensor, (optional)
                    },
                    "predictions": torch.Tensor
                }
        """
        # shape: (batch_size, channels, height, width)
        visual_features = self.visual(batch["image"])
        global_feature = visual_features[:, 0, :].unsqueeze(1)
        visual_features = visual_features[:, 1:]
        visual_features = visual_features + global_feature
# -----------------------------------------------------------------------------------------
        visual_features2 = self.visual(batch["image2"])
        global_feature2 = visual_features2[:, 0, :].unsqueeze(1)
        visual_features2 = visual_features2[:, 1:]
        visual_features2 = visual_features2 + global_feature2


        visual_features3 = self.visual(batch["image3"])
        global_feature3 = visual_features3[:, 0, :].unsqueeze(1)
        visual_features3 = visual_features3[:, 1:]
        visual_features3 = visual_features3 + global_feature3

        # concat global-feature
        # visual_features_all =  torch.cat((global_feature, global_feature2), 1)
        # visual_features_all =  torch.cat((visual_features_all, global_feature3), 1)  # [bs, 3, 768]

        # concat global+local-feature in dimension 1
        # visual_features_all =  torch.cat((visual_features, visual_features2), 1)
        # visual_features_all =  torch.cat((visual_features_all, visual_features3), 1)  # [bs, 588(= 196 * 3), 768]

        # concat global+local-feature in dimension 2()
        # visual_features_all =  torch.cat((visual_features, visual_features2), 2)
        # visual_features_all =  torch.cat((visual_features_all, visual_features3), 2)  # [bs, 196, 2304(768 * 3)]

        # fusion visual features
        visual_features_weight,visual_features2_weight, visual_features3_weight  = self.fusion(visual_features, visual_features2, visual_features3) # [bs, 196]

        visual_features = visual_features * visual_features_weight
        visual_features2 = visual_features2 * visual_features2_weight
        visual_features3 = visual_features3 * visual_features3_weight

        visual_features_all =  torch.cat((visual_features, visual_features2), 1)
        visual_features_all =  torch.cat((visual_features_all, visual_features3), 1)  # [bs, 588(= 196 * 3), 768]

# ---------------------------------contrastive learning branch---------------------------------------------
        global_feature_all = torch.cat((global_feature, global_feature2), 2)
        global_feature_all = torch.cat((global_feature_all, global_feature3), 2)
        
        global_feature_all = global_feature_all.flatten(1)
        
        caption_feature = self.contrastive_bert(batch["contrastive_caption"])

        contrastive_loss = self.infoNCE(global_feature_all, caption_feature)


#-----------------------------------------------------------------------------------------
        batch_size = visual_features_all.size(0)
        
        if "caption_tokens" in batch:
            caption_tokens = batch["caption_tokens"]
            caption_lengths = batch["caption_lengths"]
            
            # shape: (batch_size, max_caption_length, vocab_size)
            output_logits, textual_feature = self.textual(  # textual_feature [bs, max_lenth_token,1024]
                visual_features_all, caption_tokens, caption_lengths
            )

            loss = self.loss(
                output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
                caption_tokens[:, 1:].contiguous().view(-1),
            )

            output_dict: Dict[str, Any] = {
                "loss": loss + contrastive_loss,
                # Single scalar per batch for logging in training script.
                "loss_components": {"contrastive learning": contrastive_loss.clone().detach(), "captioning_forward": loss.clone().detach()},
            }
            # Do captioning in backward direction if specified.
            if self.caption_backward:
                backward_caption_tokens = batch["noitpac_tokens"]

                backward_output_logits, backward_textual_feature = self.backward_textual(
                    visual_features_all, backward_caption_tokens, caption_lengths
                )

                backward_loss = self.loss(
                    backward_output_logits[:, :-1].contiguous().view(-1, self.textual.vocab_size),
                    backward_caption_tokens[:, 1:].contiguous().view(-1),
                )
                
                output_dict["loss"] += backward_loss
                # Single scalar per batch for logging in training script.
                output_dict["loss_components"].update(
                    captioning_backward=backward_loss.clone().detach(), 
                )

            if not self.training:
                # During validation (while pretraining), get best prediction
                # at every time-step.
                output_dict["predictions"] = torch.argmax(output_logits, dim=-1)
        else:
            # During inference, get beam search predictions for forward
            # model. Predictions from forward transformer will be shifted
            # right by one time-step.
            start_predictions = visual_features.new_full(
                (batch_size,), self.sos_index
            ).long()
            # Add image features as a default argument to match callable
            # signature accepted by beam search class (partial captions only).
            beam_search_step = functools.partial(
                self.beam_search_step, visual_features
            )
            all_top_k_predictions, _ = self.beam_search.search(
                start_predictions, beam_search_step
            )
            best_beam = all_top_k_predictions[:, 0, :]
            output_dict = {"predictions": best_beam}

        return output_dict

    def beam_search_step(
        self, visual_features: torch.Tensor, partial_captions: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Given visual features and a batch of (assumed) partial captions, predict
        the distribution over vocabulary tokens for next time-step. This method
        is used by :class:`~refers.utils.beam_search.AutoRegressiveBeamSearch`.

        Parameters
        ----------
        projected_visual_features: torch.Tensor
            A tensor of shape ``(batch_size, ..., textual_feature_size)``
            with visual features already projected to ``textual_feature_size``.
        partial_captions: torch.Tensor
            A tensor of shape ``(batch_size * beam_size, timesteps)``
            containing tokens predicted so far -- one for each beam. We need all
            prior predictions because our model is auto-regressive.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size * beam_size, vocab_size)`` -- output
            distribution over tokens for next time-step.
        """

        # Expand and repeat image features while doing beam search.
        batch_size, channels, height, width = visual_features.size()
        beam_size = int(partial_captions.size(0) / batch_size)
        if beam_size > 1:
            # shape: (batch_size * beam_size, channels, height, width)
            visual_features = visual_features.unsqueeze(1).repeat(1, beam_size, 1, 1, 1)
            visual_features = visual_features.view(
                batch_size * beam_size, channels, height, width
            )

        # Provide caption lengths as current length (irrespective of predicted
        # EOS/padding tokens). shape: (batch_size, )
        caption_lengths = torch.ones_like(partial_captions)
        if len(caption_lengths.size()) == 2:
            caption_lengths = caption_lengths.sum(1)
        else:
            # Add a time-step. shape: (batch_size, 1)
            partial_captions = partial_captions.unsqueeze(1)

        # shape: (batch_size * beam_size, partial_caption_length, vocab_size)
        output_logits = self.textual(
            visual_features, partial_captions, caption_lengths
        )
        # Keep features for last time-step only, we only care about those.
        output_logits = output_logits[:, -1, :]

        # Return logprobs as required by `AutoRegressiveBeamSearch`.
        # shape: (batch_size * beam_size, vocab_size)
        next_logprobs = F.log_softmax(output_logits, dim=1)

        # Set logprobs of last predicted tokens as high negative value to avoid
        # repetition in caption.
        for index in range(batch_size * beam_size):
            next_logprobs[index, partial_captions[index, -1]] = -10000

        return next_logprobs

    def log_predictions(
        self, batch: ImageCaptionBatch, tokenizer: SentencePieceBPETokenizer
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions):
            predictions_str += f"""
                Caption tokens : {" ".join(tokens.tolist())}
                Predictions (f): {" ".join(preds.tolist())}

                """
        return predictions_str


class ForwardCaptioningModel(CaptioningModel):
    r"""
    Convenient extension of :class:`~refers.models.captioning.CaptioningModel`
    for better readability: this passes ``caption_backward=False`` to super class.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        fusion : fusion,
        contrastive_bert: TextEncoder,
        textual: TextualHead,
        beam_size: int = 5,
        max_decoding_steps: int = 30,
        sos_index: int = 1,
        eos_index: int = 2,
    ):
        super().__init__(
            visual,
            fusion,
            contrastive_bert,
            textual,
            beam_size=beam_size,
            max_decoding_steps=max_decoding_steps,
            sos_index=sos_index,
            eos_index=eos_index,
            caption_backward=False,
        )


class BidirectionalCaptioningModel(CaptioningModel):
    r"""
    Convenient extension of :class:`~refers.models.captioning.CaptioningModel`
    for better readability: this passes ``caption_backward=True`` to super class.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        fusion : fusion,
        contrastive_bert: TextEncoder,
        textual: TextualHead,
        beam_size: int = 5,
        max_decoding_steps: int = 30,
        sos_index: int = 1,
        eos_index: int = 2,
    ):
        super().__init__(
            visual,
            fusion,
            contrastive_bert,
            textual,
            beam_size=beam_size,
            max_decoding_steps=max_decoding_steps,
            sos_index=sos_index,
            eos_index=eos_index,
            caption_backward=True,
        )
