import logging
import os

from allennlp.common import Params
from allennlp.data import Dataset, DatasetReader, Instance, Token
from allennlp.data.fields import SequenceLabelField, TextField
from allennlp.data.token_indexers import (SingleIdTokenIndexer, TokenCharactersIndexer,
                                          TokenIndexer)
from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import RegexpTokenizer, BlanklineTokenizer


logger = logging.getLogger(__name__)


@DatasetReader.register('conll_two_columns')
class CoNLLTwoColumnsDatasetReader(DatasetReader):
    def __init__(self, token_indexers=None, sentence_field_name='sentence',
                 tags_field_name='tags', tags_namespace='tags'):
        if token_indexers is None:
            token_indexers = {
                'words': SingleIdTokenIndexer(namespace='tokens'),
                'chars': TokenCharactersIndexer(namespace='token_chars'),
            }
        self.token_indexers = token_indexers
        self.sentence_field_name = sentence_field_name
        self.tags_field_name = tags_field_name
        self.tags_namespace = tags_namespace

    def read(self, file_path):
        logger.info('Reading instances from file %s', file_path)
        reader = TaggedCorpusReader(
            *os.path.split(file_path), sep='\t',
            word_tokenizer=RegexpTokenizer(r'\n', gaps=True),
            sent_tokenizer=BlanklineTokenizer(),
            para_block_reader=lambda s: [s.read()])
        return Dataset([self.text_to_instance(*tuple(zip(*tagged_sent)))
                        for tagged_sent in reader.tagged_sents()])

    def text_to_instance(self, sentence, tags=None):
        sent_field = TextField([Token(w) for w in sentence], self.token_indexers)
        fields = {self.sentence_field_name: sent_field}
        if tags is not None:
            fields[self.tags_field_name] = SequenceLabelField(
                tags, sent_field, label_namespace=self.tags_namespace)
        return Instance(fields)

    @classmethod
    def from_params(cls, params):
        token_indexers_params = params.pop('token_indexers', Params({}))
        token_indexers = TokenIndexer.dict_from_params(token_indexers_params)
        sentence_field_name = params.pop('sentence_field_name', 'sentence')
        tags_field_name = params.pop('tags_field_name', 'tags')
        tags_namespace = params.pop('tags_namespace', 'tags')
        params.assert_empty(cls.__name__)
        return cls(token_indexers=token_indexers, sentence_field_name=sentence_field_name,
                   tags_field_name=tags_field_name, tags_namespace=tags_namespace)
