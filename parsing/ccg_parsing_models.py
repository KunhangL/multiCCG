import sys
from typing import List
import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.append('..')
from ccg_supertagger.models import BaseSupertaggingModel
from ccg_supertagger.supertagger import CCGSupertagger


SupertaggingRepresentations = torch.Tensor  # l_sent*supertagging_n_classes

DATA_MASK_PADDING = 0


class BaseParsingModel(nn.Module):

    def __init__(
        self,
        model_dir: str,
        supertagging_n_classes: int,
        embed_dim: int,
        checkpoint_path: str,
        device: torch.device = torch.device('cuda:0')
    ):
        super().__init__()
        self.device = device
        self.supertagger = CCGSupertagger(
            model=BaseSupertaggingModel(
                model_dir, supertagging_n_classes, embed_dim
            ),
            tokenizer=AutoTokenizer.from_pretrained(model_dir),
            device=self.device
        )
        self.supertagger._load_model_checkpoint(checkpoint_path)

    def forward(
        self,
        pretokenized_sents: List[List[str]]
    ) -> List[SupertaggingRepresentations]:
        """
        Outputs:
            the embedding of each word in every sentence
            from the base supertagging model
        """
        return self.supertagger.get_model_outputs_for_batch(pretokenized_sents)