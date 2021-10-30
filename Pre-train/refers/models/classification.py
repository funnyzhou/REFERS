from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from refers.data.structures import Batch
from refers.data.tokenizers import SentencePieceBPETokenizer
from refers.modules.textual_heads import TextualHead
from refers.modules.visual_backbones import VisualBackbone


class ClassificationModel(nn.Module):
    r"""
    A model to perform classification (generally, with multiple targets). It is
    composed of a :class:`~refers.modules.visual_backbones.VisualBackbone` and a
    :class:`~refers.modules.textual_heads.TextualHead` on top of it.

    .. note::

        As with currently available textual heads, only one textual head is
        supported here: :class:`~refers.modules.textual_heads.LinearTextualHead`.

    During training, it minimizes the KL-divergence loss with a K-hot vector,
    with values ``1/K``, where K are the number of unique labels to classify.

    Parameters
    ----------
    visual: refers.modules.visual_backbones.VisualBackbone
        A :class:`~refers.modules.visual_backbones.VisualBackbone` which
        computes visual features from an input image.
    textual: refers.modules.textual_heads.TextualHead
        A :class:`~refers.modules.textual_heads.TextualHead` which
        makes final predictions conditioned on visual features.
    ignore_indices: List[int]
        Ignore a set of token indices while computing KL-divergence loss. These
        are usually the special tokens such as ``[SOS]``, ``[EOS]`` etc.
    """

    def __init__(
        self,
        visual: VisualBackbone,
        textual: TextualHead,
        ignore_indices: List[int],
    ):
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.ignore_indices = ignore_indices

    def forward(self, batch: Batch):
        r"""
        Given a batch of images and set of labels (named "caption_tokens" just
        for the sake of simplicity), perform classification with multiple
        targets by minimizing a KL-divergence loss.

        Parameters
        ----------
        batch: refers.data.structures.ImageCaptionBatch
            A batch of images and label set (as key named "caption_tokens").

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
                        "classification": torch.Tensor,
                    },
                    "predictions": torch.Tensor
                }
        """

        # shape: (batch_size, visual_feature_size, ...)
        visual_features = self.visual(batch["image"])
        batch_size = visual_features.size(0)

        # Get logits and further log-probabilities.
        # shape: (batch_size, vocab_size)
        logits = self.textual(visual_features)
        logprobs = F.log_softmax(logits, dim=1)

        # Average log-probs per unique token in associated caption to compute
        # loss. This is simply cross-entropy with target-vector as a K-hot
        # vector. Do in a for-loop, there isn't a straightforward vectorized way.
        loss = torch.tensor(0.0, device=logprobs.device)

        for index in range(batch_size):
            # Get unique tokens for particular instance.
            unique_tokens = batch["caption_tokens"][index].unique()

            # Ignore indices of special tokens such as [SOS], [EOS] etc. and
            # any other token specified.
            unique_tokens = [
                t for t in unique_tokens if t not in self.ignore_indices
            ]
            # Get log-probabilities corresponding to these tokens.
            instance_logprobs = logprobs[index, unique_tokens].mean()

            # Accumulate negative log-probability for this instance in loss.
            loss = loss - instance_logprobs

        # Average loss across instances.
        output_dict: Dict[str, Any] = {"loss": loss / batch_size}

        # Single scalar per batch for logging to tensorboard in training script.
        output_dict["loss_components"] = {
            "classification": loss.clone().detach() / batch_size
        }
        # Return top-10 tokens according to log-probabilities during validation.
        # Useful for logging.
        if not self.training:
            top_logprobs, top_tokens = logprobs.topk(k=10, dim=1)
            output_dict["predictions"] = top_tokens

        return output_dict


class TokenClassificationModel(ClassificationModel):
    r"""
    Convenient extension of :class:`~refers.models.classification.ClassificationModel`
    for better readability (this only modifies the tensorboard logging logic).

    Ground truth targets here are a set of unique caption tokens (ignoring the
    special tokens like ``[SOS]``, ``[EOS]`` etc.).
    """

    def log_predictions(
        self, batch: Batch, tokenizer: SentencePieceBPETokenizer
    ) -> str:

        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions):
            # Predictions here are individual tokens, and do not have any order
            # like captions, so decode them separately so we don't strip off
            # metaspace character and special tokens if any.
            preds = [tokenizer.id_to_token(p) for p in preds.tolist()]
            predictions_str += f"""
                Caption tokens : {tokenizer.decode(tokens.tolist())}
                Predictions (f): {" ".join(preds)}

                """
        return predictions_str


class MultiLabelClassificationModel(ClassificationModel):
    r"""
    Convenient extension of :class:`~refers.models.classification.ClassificationModel`
    for better readability (this only modifies the tensorboard logging logic).

    Ground truth targets here are a set of unique instances in images (ignoring
    the special background token, category id = 0 in COCO).
    """

    def log_predictions(
        self, batch: Batch, tokenizer: SentencePieceBPETokenizer = None,
    ) -> str:
        # We accept `tokenizer` for having consistent API but don't use it here.
        self.eval()
        with torch.no_grad():
            predictions = self.forward(batch)["predictions"]
        self.train()

        predictions_str = ""
        for tokens, preds in zip(batch["caption_tokens"], predictions):
            # Predictions here are COCO category IDs, let them be as is.
            # Sorted ground truth, remove background tokens.
            tokens = sorted([t for t in tokens.tolist() if t != 0])
            preds = sorted(preds.tolist()[:len(tokens)])
            predictions_str += f"""
                COCO Instance IDs (GT)   : {tokens}
                COCO Instance IDs (Pred) : {preds}

                """
        return predictions_str
