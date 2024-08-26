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

#Code by Arina Belova

from transformers import BertForTokenClassification
import torch
from transformers import BertTokenizer
import numpy as np
import nltk.data

nltk.download('punkt')

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from transformers import BertForTokenClassification, AdamW

from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report

import torch.nn as nn

from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize
from utils.spec_tokenizers import custom_word_tokenize
from nltk.tokenize.util import align_tokens
import os


class NER_BERT(object):
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag2idx = {'O':0, 'ID':1, 'PHI':2, 'NAME':3, 'CONTACT':4, 'DATE':5, 'AGE':6, 'PROFESSION':7, 'LOCATION':8, 'PAD': 9}
    tag_values = ["O","ID", "PHI", "NAME", "CONTACT", "DATE", "AGE", "PROFESSION", "LOCATION", "PAD"]

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', num_labels=len(tag2idx), do_lower_case=False)

    MAX_LEN = 75  # max length of sequence, needs for padding
    bs = 32 # batch size

    """Abstract class that other NER plugins should implement"""

    def __init__(self):
        if os.path.exists("Models/NER_BERT.pt"):
            print("Loading model")
            state_dict = torch.load("Models/NER_BERT.pt", map_location=torch.device('cpu'))
            print("Loaded model")

            self.model = BertForTokenClassification.from_pretrained(
                "bert-base-cased",
                state_dict=state_dict,
                num_labels=len(NER_BERT.tag2idx),
                output_attentions=False,
                output_hidden_states=True
            )
        else:
            self.model = BertForTokenClassification.from_pretrained(
                "bert-base-cased",
                num_labels=len(NER_BERT.tag2idx),
                output_attentions=False,
                output_hidden_states=True
            )

    def perform_NER(self, text):
        """Implementation of the method that should perform named entity recognition"""
        # tokenizer to divide data into sentences (thanks, nltk)

        list_of_sents = sent_tokenize(text)
        list_of_tuples_by_sent = []

        # When sent exceeds max_length in output, second part of the sentence gets lost after truncation
        # This is to format the output of the perform_NER in a way that all tokens in text have labels,
        # so they are comparable to the other algs when union/intersection
        full_text_tokenized = custom_word_tokenize(text,use_bert_tok= True)
        spans = align_tokens(full_text_tokenized, text)
        n = 0
        point = 0 
        
        for sent in list_of_sents:
            tokenized_sentence = self.tokenizer.encode(sent,
                                                       truncation=True)  # BERT tokenizer is clever, it will internally divide the sentence by words, so all we need to provide there is sentence and it will return an array where each token is either special token/word/subword, refer to BERT WordPiece tokenizer approach
            # truncation=True to comply with 512 length of the sentence
            input_ids = torch.tensor([tokenized_sentence])

            with torch.no_grad():
                # Run inference/classification
                output = self.model(input_ids)
            label_indices = np.argmax(output[0].to("cpu").numpy(), axis=2)
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
            new_tokens, new_labels, new_start, new_end = [], [], [], []
            for token, label_idx in zip(tokens, label_indices[0]):
                # remove [CLS] and [SEP] tokens to comply wth xml structure and
                # remove quotation to match custom_word_tokenization and make the output comparable to other algs when union
                if (token == '[CLS]') or (token == '[SEP]') or (token == "'") or (token == "\"") or (token == "`"):
                    continue
                elif token.startswith("##"):
                    new_tokens[-1] = new_tokens[-1] + token[2:]
                else:
                    new_labels.append(self.tag_values[label_idx])
                    new_tokens.append(token)
            
            new_start, new_end = zip(*align_tokens(new_tokens, text[point:]))
            new_start, new_end = [point + st for st in new_start], [point + en for en in new_end]

            len_tok_sent = len(custom_word_tokenize(sent, use_bert_tok= True))
            seq_toks = full_text_tokenized[n : n + len_tok_sent]
            seq_start, seq_end = zip(*spans[n : n + len_tok_sent])
            seq_labels = ['O'] * len_tok_sent
            ori_seq = list(zip(seq_toks, seq_start, seq_end))
           
            list_of_tuples = list(zip(seq_toks, seq_labels))
            for token, label, st, en in zip(new_tokens, new_labels, new_start, new_end):
                if tuple([token,st,en]) in ori_seq:
                    idx = ori_seq.index(tuple([token,st,en]))
                    list_of_tuples[idx] = tuple([token,label])
            n = n + len_tok_sent
            
            list_of_tuples_by_sent.append(list_of_tuples)

            point = point + len(sent)

        return list_of_tuples_by_sent

        # Needed for transform_sequences

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []
        for word, label in zip(sentence, text_labels):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = NER_BERT.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)
            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)
            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)
        return tokenized_sentence, labels

    def transform_sequences(self, tokens_labels):
        """method that transforms sequences of (token,label) into feature sequences. Returns two sequence lists for X and Y"""
        print("I am in transform seq")

        tokenized_sentences = []
        labels = []
        for index, sentence in enumerate(tokens_labels):
            text_labels = []
            sentence_to_feed = []
            for word_label in sentence:
                text_labels.append(word_label[1])
                sentence_to_feed.append(word_label[0])
            a, b = self.tokenize_and_preserve_labels(sentence_to_feed, text_labels)
            tokenized_sentences.append(a)
            labels.append(b)

        input_ids = pad_sequences([NER_BERT.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_sentences],
                                  maxlen=NER_BERT.MAX_LEN, dtype="long", value=0.0,
                                  truncating="post", padding="post")

        tags = pad_sequences([[NER_BERT.tag2idx.get(l) for l in lab] for lab in labels],
                             maxlen=NER_BERT.MAX_LEN, value=NER_BERT.tag2idx["PAD"], padding="post",
                             dtype="long", truncating="post")

        # Result is pair X (array of sentences, where each sentence is an array of words) and Y (array of labels)
        return input_ids, tags

    def learn(self, X_train, Y_train, epochs=1):
        """Function that actually train the algorithm"""
        # if torch.cuda.is_available():
        #     self.model.cuda()

        tr_masks = [[float(i != 0.0) for i in ii] for ii in X_train]

        print("READY TO CREATE SOME TENZORS!!!!!!!!!!!!!!!!!!!!!!!!!!")
        tr_inputs = torch.tensor(X_train).type(torch.long)
        tr_tags = torch.tensor(Y_train).type(torch.long)
        tr_masks = torch.tensor(tr_masks).type(torch.long)

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=NER_BERT.bs)

        print("READY TO PREPARE OPTIMIZER!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Weight decay in Adam optimiser (adaptive gradient algorithm) is a regularisation technique which is extensively disucssed in this paper:
        # https://arxiv.org/abs/1711.05101
        # (Like L2 for SGD but different)
        # resularisation of the model objective function in order to prevent overfitting of the model.
        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},  # in AdamW implementation (default: 1e-2)
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        # TODO: change to new implementation of AdamW: torch.optim.AdamW(...)
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8
        )

        max_grad_norm = 1.0

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        print("START TRAINING!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        ## Store the average loss after each epoch so we can plot them.
        loss_values, validation_loss_values = [], []

        for _ in trange(epochs, desc="Epoch"):
            # ========================================
            #               Training
            # ========================================
            # Perform one full pass over the training set.
            # clean the cache not to fail with video memory
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            # Put the model into training mode.
            self.model.train()
            # Reset the total loss for this epoch.
            total_loss = 0

            # Training loop
            for step, batch in enumerate(train_dataloader):
                # add batch to gpu
                batch = tuple(b.to(NER_BERT.device) for b in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # Always clear any previously calculated gradients before performing a backward pass.
                self.model.zero_grad()
                # forward pass
                # This will return the loss (rather than the model output)
                # because we have provided the `labels`.
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask, labels=b_labels)
                # get the loss
                loss = outputs[0]

                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # track train loss
                total_loss += loss.item()
                # Clip the norm of the gradient
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)
                # update parameters
                optimizer.step()
                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)
            print("Average train loss: {}".format(avg_train_loss))

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            # Plot the learning curve.
            plt.figure()
            plt.plot(loss_values, 'b-o', label="training loss")
            # Label the plot.
            plt.title("Learning curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

            plt.show()

    def evaluate(self, X_test, Y_test):
        """Function to evaluate algorithm"""
        val_masks = [[float(i != 0.0) for i in ii] for ii in X_test]
        val_inputs = torch.tensor(X_test).type(torch.long)
        val_tags = torch.tensor(Y_test).type(torch.long)
        val_masks = torch.tensor(val_masks).type(torch.long)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=NER_BERT.bs)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode to set dropout and batch normalization layers to evaluation mode to have consistent results
        self.model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [NER_BERT.tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if NER_BERT.tag_values[l_i] != "PAD"]
        valid_tags = [NER_BERT.tag_values[l_i] for l in true_labels
                      for l_i in l if NER_BERT.tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(valid_tags, pred_tags, average='weighted')))
        labels = ["ID", "PHI", "NAME", "CONTACT", "DATE", "AGE",
                  "PROFESSION", "LOCATION"]
        print(classification_report(valid_tags, pred_tags, digits=4, labels=labels))
        print()

        # # Use plot styling from seaborn.
        # sns.set(style='darkgrid')

        # # Increase the plot size and font size.
        # sns.set(font_scale=1.5)
        # plt.rcParams["figure.figsize"] = (12,6)

        # # Plot the learning curve.
        # plt.plot(loss_values, 'b-o', label="training loss")
        # plt.plot(validation_loss_values, 'r-o', label="validation loss")

        # # Label the plot.
        # plt.title("Learning curve")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.legend()

        # plt.show()

    def save(self, model_path):
        """
        Function to save model. Models are saved as h5 files in Models directory. Name is passed as argument
        :param model_path: Name of the model file
        :return: Doesn't return anything
        """
        torch.save(self.model.state_dict(), "Models/" + model_path + ".pt")
        print("Saved model to disk")
