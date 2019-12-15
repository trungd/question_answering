import csv
import json
import os
import subprocess
from typing import Dict, List

from dlex.torch.utils.ops_utils import maybe_cuda
from dlex.torch.utils.variable_length_tensor import pad_sequence
from dlex.utils import logger
from dlex_impl.question_answering.src.datasets.squad.builder import detokenize
from torch import LongTensor
from tqdm import tqdm

from ..base import QADataset, QABatch


class PytorchSQuAD_V1(QADataset):
    vocab_word = None
    vocab_char = None
    word_embedding_layer = None

    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)
        data = []

        with open(self.builder.output_file_path(mode)) as f:
            reader = csv.reader(f)
            examples = list(reader)

        examples = [[val.split(' ') for val in ex] for ex in examples]
        if mode == 'train':
            examples = list(filter(
                lambda ex: len(ex[1]) <= self.configs.paragraph_max_length and len(ex[2]) <= self.configs.question_max_length,
                examples))

        # We want these variables to be shared among instances
        PytorchSQuAD_V1.word_embedding_layer, vocab_word = self.load_embeddings(
            tokens=self.builder.vocab_word.tolist(),
            specials=['<sos>', '<eos>', '<oov>', '<pad>'])
        PytorchSQuAD_V1.vocab_word = vocab_word or self.builder.vocab_word
        PytorchSQuAD_V1.vocab_char = self.builder.vocab_char

        for id, context, question, answer_span in tqdm(examples, desc="Loading data (%s)" % mode):
            data.append(dict(
                id=id[0],
                context=context,
                cw=self.vocab_word.encode_token_list(context),
                qw=self.vocab_word.encode_token_list(question),
                cc=[self.vocab_char.encode_token_list(list(w)) for w in context],
                qc=[self.vocab_char.encode_token_list(list(w)) for w in question],
                answer_span=[int(pos) for pos in answer_span[0].split('-')]
            ))
        self._data = data

    @property
    def word_dim(self):
        return self.configs.embeddings.dim

    @property
    def vocab_size(self):
        if PytorchSQuAD_V1.vocab_word:
            return len(PytorchSQuAD_V1.vocab_word)
        else:
            return len(self.builder.vocab_word)

    @property
    def char_dim(self):
        return self.configs.char_dim

    def collate_fn(self, batch: List[Dict]):
        # batch.sort(key=lambda item: len(item.X), reverse=True)
        w_contexts = [item['cw'] for item in batch]
        w_questions = [item['qw'] for item in batch]

        # char_max_length = max([max(len(c) for c in item['cc']) for item in batch])
        char_max_length = 16
        c_contexts = [[
            char_idx[:char_max_length] + max(char_max_length - len(char_idx), 0) * [self.vocab_char.blank_token_idx]
            for char_idx in item['cc']
        ] for item in batch]

        char_max_length = max([max(len(c) for c in item['qc']) for item in batch])
        c_questions = [[
            char_idx + (char_max_length - len(char_idx)) * [self.vocab_char.blank_token_idx]
            for char_idx in item['qc']
        ] for item in batch]

        answer_spans = LongTensor([item['answer_span'] for item in batch])

        w_contexts, w_context_lengths = pad_sequence(w_contexts, padding_value=self.vocab_word.blank_token_idx)
        w_questions, w_question_lengths = pad_sequence(w_questions, padding_value=self.vocab_word.blank_token_idx)
        c_contexts, _ = pad_sequence(c_contexts, padding_value=self.vocab_char.blank_token_idx)
        c_questions, _ = pad_sequence(c_questions, padding_value=self.vocab_char.blank_token_idx)

        batch_x = super().collate_fn(list(zip(
                w_contexts, w_context_lengths, c_contexts, w_questions, w_question_lengths, c_questions)))
        return QABatch(
            X=[maybe_cuda(x) for x in batch_x],
            Y=maybe_cuda(answer_spans))

    def evaluate(self, y_pred, y_ref, metric: str, output_path: str):
        if metric in {"em", "f1"}:
            # use official evaluation script
            assert len(y_pred) == len(self.data)
            ret = {}
            for pred, data in zip(y_pred, self.data):
                ans = ' '.join(data['context'][pred[0]:pred[1] + 1])
                ans = detokenize(ans)
                ret[data['id']] = ans

            f_name = output_path + '.json'
            with open(f_name, 'w') as f:
                json.dump(ret, f, indent=2)
                logger.debug("Results saved to %s" % f_name)

            process = subprocess.Popen([
                'python', os.path.join(*os.path.split(os.path.realpath(__file__))[:-1], 'evaluate-v1.1.py'),
                os.path.join(self.builder.get_working_dir(), "%s-v1.1.json" % self.mode),
                f_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, _ = process.communicate()

            out = json.loads(out.decode())
            return out['exact_match'] if metric == 'em' else out['f1']
        else:
            super().evaluate(y_pred, y_ref, metric, output_path)

    def format_output(self, y_pred, batch_input) -> (str, str, str):
        start, end = y_pred
        return (
            ' '.join(self.vocab_word.decode_idx_list(batch_input.X[3])),
            ' '.join(self.vocab_word.decode_idx_list(batch_input.X[0][batch_input.Y[0]:batch_input.Y[1] + 1])),
            ' '.join(self.vocab_word.decode_idx_list(batch_input.X[0][start:end + 1]))
        )