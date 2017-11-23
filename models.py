from allennlp.common import Params
from allennlp.models import Model
from allennlp.modules import (ConditionalRandomField as CRF, Seq2SeqEncoder, TextFieldEmbedder,
                              TimeDistributed)
from allennlp.nn.initializers import Initializer, InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import Metric, SpanBasedF1Measure
from torch.nn import Linear


@Model.register('bilstm_seq_tagger')
class BiLSTMCRFSequenceTagger(Model):
    def __init__(self, vocab, text_field_embedder, hidden_size=128, num_layers=2, dropout=0.5,
                 tag_namespace='tags', initializer=None, metric=None):
        if initializer is None:
            initializer = InitializerApplicator()
        if metric is None:
            metric = SpanBasedF1Measure(vocab, tag_namespace=tag_namespace)

        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.tag_namespace = tag_namespace
        self.initializer = initializer
        self.metric = metric
        self.seq2seq_encoder = Seq2SeqEncoder.from_params(Params({
            'type': 'lstm',
            'input_size': text_field_embedder.get_output_dim(),
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'batch_first': True,
            'bidirectional': True,
        }))
        self.num_tags = vocab.get_vocab_size(tag_namespace)
        self.tags_projection_layer = TimeDistributed(
            Linear(self.seq2seq_encoder.get_output_dim(), self.num_tags))
        self.crf = CRF(self.num_tags)
        self.initializer(self)

    def forward(self, sentence, tags=None):
        """Forward computation.

        Arguments
        ---------
        sentence : Dict[str, Variable[torch.LongTensor]]
            Mapping from indexer name to a tensor of indices. The indices tensor can
            have shape like ``(batch_size, num_tokens)`` if indexed by tokens or
            ``(batch_size, num_tokens, num_chars)`` if indexed by characters.
        tags : Variable[torch.LongTensor]
            Tag indices for this batch. This should have a shape ``(batch_size, num_tokens)``.

        Returns
        -------
        output : Dict[str, Variable]
            Output dictionary with keys ``logits``, ``mask``, and ``loss``.
        """
        embedded = self.text_field_embedder(sentence)  # (bsize, n_tokens, emb_dim)
        encoded = self.seq2seq_encoder(embedded)  # (bsize, n_tokens, out_dim)
        logits = self.tags_projection_layer(encoded)  # (bsize, n_tokens, n_tags)
        mask = get_text_field_mask(sentence)
        output = {'logits': logits, 'mask': mask}
        if tags is not None:
            llh = self.crf.forward(logits, tags, mask=mask)
            output['loss'] = -llh
            self.metric(logits, tags, mask=mask)
        return output

    def decode(self, output):
        """Compute best tag sequence.

        Arguments
        ---------
        output : Dict[str, Variable]
            Output dictionary returned by ``.forward()``.

        Returns
        -------
        output : Dict[str, Variable]
            The same dictionary given as input but updated with keys ``predicted_tags``
            and ``prediction_probs``.
        """
        predicted_tags = self.crf.viterbi_tags(output['logits'], output['mask'])
        prediction_probs = output['logits'] * 0.
        for i, sentence_tags in enumerate(predicted_tags):
            for j, tag_id in enumerate(sentence_tags):
                prediction_probs[i, j, tag_id] = 1.
        output.update({'predicted_tags': predicted_tags, 'prediction_probs': prediction_probs})
        return output

    def get_metrics(self, reset=False):
        return self.metric.get_metric(reset)

    @classmethod
    def from_params(cls, vocab, params):
        text_field_embedder = TextFieldEmbedder.from_params(
            vocab, params.pop('text_field_embedder'))
        hidden_size = params.pop('hidden_size', 128)
        num_layers = params.pop('num_layers', 2)
        dropout = params.pop('dropout', 0.5)
        tag_namespace = params.pop('tag_namespace', 'tags')
        initializer = Initializer.from_params(params.pop('initializer', Params({})))
        metric = Metric.from_params(params.pop('metric', Params({})), vocab=vocab)
        return cls(vocab, text_field_embedder, hidden_size=hidden_size, num_layers=num_layers,
                   dropout=dropout, tag_namespace=tag_namespace, initializer=initializer,
                   metric=metric)
