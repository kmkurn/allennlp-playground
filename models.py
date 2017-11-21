from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN
from allennlp.models import Model
from allennlp.modules import ConditionalRandomField as CRF


@Model.register('bilstm_seq_tagger')
class BiLSTMSequenceTagger(Model):
    def __init__(self, vocab, text_field_embedder, seq2seq_encoder, words_namespace='tokens',
                 tags_namespace='tags'):
        self.vocab = vocab
        self.text_field_embedder = text_field_embedder
        self.seq2seq_encoder = seq2seq_encoder
        self.words_namespace = words_namespace
        self.tags_namespace = tags_namespace

    def forward(self, words, tags=None):
        """Forward computation.

        Arguments
        ---------
        words : Dict[str, torch.LongTensor]
            Mapping from indexer name to a tensor of indices. The indices tensor can
            have shape like ``(batch_size, num_tokens)`` if indexed by tokens or
            ``(batch_size, num_tokens, num_chars)`` if indexed by characters.
        tags : torch.LongTensor
            Tag indices for this batch. This should have a shape ``(batch_size, num_tokens)``.
        """
        embedded_words = self.text_field_embedder(words)  # (bsize, n_tokens, emb_dim)
        encoded = self.seq2seq_encoder(embedded_words)  # (bsize, n_tokens, out_dim)
        num_tags = self.vocab.get_vocab_size(self.tags_namespace)
        padding_idx = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN,
                                                 namespace=self.words_namespace)
