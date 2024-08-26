"""
Copyright 2020 ICES, University of Manchester, Evenset Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#Code by Nikola Milosevic
import nltk
nltk.download('punkt')
from nltk.tokenize.util import align_tokens
from nltk.tokenize.treebank import TreebankWordTokenizer
import re
import tensorflow_hub as hub
#from bert.tokenization import FullTokenizer
import tensorflow as tf
from transformers import BertTokenizer
sess = tf.compat.v1.Session()

_treebank_word_tokenizer = TreebankWordTokenizer()
_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def tokenize_to_seq(documents):
    sequences = []
    sequence = []
    for doc in documents:
        if len(sequence)>0:
            sequences.append(sequence)
        sequence = []
        text = doc["text"]
        file = doc["id"]
        tokens = custom_span_tokenize(text)
        for token in tokens:
            token_txt = text[token[0]:token[1]]
            found = False
            for tag in doc["tags"]:
                if int(tag["start"])<=token[0] and int(tag["end"])>=token[1]:
                    token_tag = tag["tag"]
                    #token_tag_type = tag["type"]
                    found = True
            if found==False:
                token_tag = "O"
                #token_tag_type = "O"
            sequence.append((token_txt,token_tag))
            if token_txt == "." or token_txt == "? " or token_txt == "!":
                sequences.append(sequence)
                sequence = []
        sequences.append(sequence)
    return sequences


def tokenize_fa(documents, use_bert_tok = False):
    """
              Tokenization function. Returns list of sequences

              :param documents: list of texts
              :type language: list

              """
    sequences = []
    for doc in documents:
        text = doc
        sequences = custom_word_tokenize(text, use_bert_tok = use_bert_tok, incl_tok_sent = True)
        sequences = [list(zip(seq, ['O'] * len(seq))) for seq in sequences]
    return sequences

# This function is defined to manage quotations and 
# usred as aux to get CRF, dictionaries and MedCAT outputs in format 
# that is comparable to BERT outputs when UNION

def aux_bert_tokenization(sent):
    tokens = BertTokenizer.tokenize(_bert_tokenizer, sent)
    new_tokens = []
    for tkn in tokens:    
        if tkn.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + tkn[2:]
        else:
            new_tokens.append(tkn)
    # new_tokens = [x for x in new_tkn if x != "'" ]
    return new_tokens


def custom_span_tokenize(text, language='english', preserve_line=True, use_bert_tok = False, incl_tok_sent = False):
    """
            Returns a spans of tokens in text.

            :param text: text to split into words
            :param language: the model name in the Punkt corpus
            :type language: str
            :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
            :type preserver_line: bool
            """
    tokens = custom_word_tokenize(text, use_bert_tok = use_bert_tok, incl_tok_sent = incl_tok_sent)
    tokens = ['"' if tok in ['``', "''"] else tok for tok in tokens]

    return list(map(lambda x : align_tokens(x, text), tokens)) if incl_tok_sent else align_tokens(tokens, text)

def custom_word_tokenize(text, language='english', preserve_line=False, use_bert_tok = False, incl_tok_sent = False):
    """
    Return a tokenized copy of *text*,
    using NLTK's recommended word tokenizer
    (currently an improved :class:`.TreebankWordTokenizer`
    along with :class:`.PunktSentenceTokenizer`
    for the specified language).

    :param text: text to split into words
    :param text: str
    :param language: the model name in the Punkt corpus
    :type language: str
    :param preserve_line: An option to keep the preserve the sentence and not sentence tokenize it.
    :type preserver_line: bool
    """
    tokens = []
    sentences = [text] if preserve_line else nltk.sent_tokenize(text, language)
    tokenized_sentences = []
    for sent in sentences:
        if use_bert_tok:
            ## Using bert tokenizer
            toks = aux_bert_tokenization(sent)
            toks = [x for x in toks if (x != "'") and (x != "\"") and (x != "`") ]
            tokens.extend(toks)
            tokenized_sentences.append(toks)
            #This is needed for the align_tokens in custom_span
        else: ## treeBankTokenizer (the default tokenizer) with some adjustments
            _sent = []
            for wrd in sent.split():
                if "'" in wrd:
                    _sent.extend(aux_bert_tokenization(wrd))
                else:
                    _sent.append(wrd)
            _sent = " ".join(_sent)           
            # this part is to handdle quotations
            toks = _treebank_word_tokenizer.tokenize(_sent)
            new_toks = []
            for tok_idx, tok in enumerate(toks):
                if ("'" in tok) or ("\"" in tok) or ("`" in tok):
                    new_toks.extend(aux_bert_tokenization(tok))
                else:
                    new_toks.append(tok)
            new_toks = [x for x in new_toks if (x != "'") and (x != "\"") and (x != "`")]   
            new_sent = []
            for token in new_toks:
                if "-" in token: 
                    m = re.compile("(\d+)(-)([a-zA-z-]+)")
                    g = m.fullmatch(token)
                    if g:
                        for group in g.groups():
                            tokens.append(group)
                            new_sent.append(group)
                    else:
                        tokens.append(token)
                        new_sent.append(token)
                else:
                    tokens.append(token)
                    new_sent.append(token)
            tokenized_sentences.append(new_sent)
    output = tokenized_sentences if incl_tok_sent else tokens
    return output

def shape(self,word):
    shape = ""
    for letter in word:
        if letter.isdigit():
            shape = shape + "d"
        elif letter.isalpha():
            if letter.isupper():
                shape = shape + "W"
            else:
                shape = shape + "w"
        else:
            shape = shape + letter
    return shape


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []
    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = aux_bert_tokenization(word)
        n_subwords = len(tokenized_word)
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)
        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)
    return tokenized_sentence, labels
