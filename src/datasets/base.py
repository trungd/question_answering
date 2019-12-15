from dataclasses import dataclass

import torch
from dlex.datasets.nlp.torch import NLPDataset
from dlex.torch.datatypes import Batch, BatchItem


class QADataset(NLPDataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)


@dataclass
class BatchX:
    context_word: torch.Tensor
    context_word_lengths: torch.LongTensor
    context_char: torch.Tensor
    question_word: torch.Tensor
    question_word_lengths: torch.LongTensor
    question_char: torch.Tensor


class QABatch(Batch):
    X: BatchX
    Y = None

    def __len__(self):
        return len(self.Y)

    def item(self, i: int) -> BatchItem:
        return BatchItem(
            X=[x[i] for x in self.X],
            Y=self.Y[i].cpu().detach().numpy())