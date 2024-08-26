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
"""
    *evaluate.py* - Evaluate algorithm on test data from i2b2
    IMPORTANT: you need to have i2b2 data in order to evaluate, as algoithm works for now only with the i2b2 layout data
    Example with one algorithm eval metric: python evaluate.py --source_location "../testing-PHI-Gold-fixed/" --algorithm NER_BERT
    Example with 2 algorithms union/intersection to provide: python evaluate.py --source_location "../testing-PHI-Gold-fixed/" --algorithm1 NER_BERT --algorithm2 NER_CRF_dictionaries --resolution union
    Code by: Arina Belova
"""

import argparse
import importlib

from utils.readers import read_i2b2_data
import utils.spec_tokenizers
from seqeval.metrics import accuracy_score
from sklearn.metrics import f1_score, classification_report
import re

if __name__ == "__main__":

    """
    Evaluates algorithm of selection on test data
    """

    print("Evaluating framework")
    parser = argparse.ArgumentParser(description='Evaluation framework for Named Entity recognition')
    parser.add_argument('--source_location', help='source location of the dataset on your hard disk')
    parser.add_argument('--algorithm', help='algorithm to use')
    parser.add_argument('--algorithm1', help='first algorithm to compare if having more than one algorithm. This one is leading algorithm!!!!')
    parser.add_argument('--algorithm2', help='second algorithm to compare if having more than one algorithm')
    parser.add_argument('--resolution', help='union/intersection')
    #parser.add_argument('--lead', help="leading algorithm to resolve disputes about token's label")

    args = parser.parse_args()

    # Beginning of data preparation
    #####################################################################################################

    path_to_data = args.source_location
    # path_to_alg = args.algorithm_location
    documents = read_i2b2_data(path_to_data) # id, text, tags fields in this dictionary
    if documents== None:
        print("Error: No input source is defined")
        exit(2)

    tokens_labels = utils.spec_tokenizers.tokenize_to_seq(documents)

#####################################################################################################
# This is simply needed for getting well-disributed separated by punctuation signs tokens and labels for the double-algos evaluation.

    final_sequences, fin_tokenized_sentence, fin_labels = [], [], []

    for sequence in tokens_labels:
        tokenized_sentence = []
        labels = []
        for tup in sequence:
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = re.split("([\W | _])", tup[0]) # the most fucking clever regexpr in my life
            tokenized_word = [t  for t in tokenized_word if t != ""]
            n_subwords = len(tokenized_word)
            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)
            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([tup[1]] * n_subwords)
        fin_tokenized_sentence.append(tokenized_sentence)
        fin_labels.append(labels)

    for i in range(0,len(fin_labels)):
        sentence = []
        for j in range(0,len(fin_labels[i])):
            sentence.append((fin_tokenized_sentence[i][j],fin_labels[i][j]))

        final_sequences.append(sentence)


    ground_truth = []
    ground_tokens = []
    for sequence in final_sequences:
        for t, l in sequence:
            #print("{}\t{}".format(t,l))
            ground_truth.append(l)
            ground_tokens.append(t)
    #print("GROUND TOKENS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # for t, l in zip(ground_tokens, ground_truth):
    #     print(f"{t}\t{l}")
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # WE DO NOT USE THOSE FINAL_SEQUENCES IN MODELS!!!!!
    # End of data preparation
    #####################################################################################################

    # Algorithm 1 is a leading algorithm
    algorithm = args.algorithm
    algorithm1 = args.algorithm1
    algorithm2 = args.algorithm2
    resolution = args.resolution

    # for token in tokens_labels:
    #     for t in token:
    #         print("{}\t{}".format(t[0],t[1]))

##########################################################################################################
# Two-algorithms case
    if algorithm1 != None and algorithm2 != None:
        package1 = "ner_plugins."+ algorithm1
        package2= "ner_plugins."+ algorithm2
        inpor1 = importlib.import_module(package1)
        inpor2 = importlib.import_module(package2)
        class1_ = getattr(inpor1, algorithm1)
        class2_ = getattr(inpor2, algorithm2)

        instance1 = class1_()

        # print("TOKEN LABELS!!!!!!!!!!!!!!!!!")
        # print(tokens_labels)
        # print("GOLD TOKENS!!!!!!!!!!!!!!!")
        # print(ground_tokens)
        # print("GOLD LABELS!!!!!!!!!!!!!!!")
        # print(ground_truth)
        # print("TOKENS LABELS LENGTH")
        # count = 0
        # for sent in tokens_labels:
        #     count += len(sent)
        # print(count)

        X1, Y1 = instance1.transform_sequences(tokens_labels)

        # print("LENGTH OF FATURIZED REPRESENTATION")
        # count = 0
        # for sent in X1:
        #     count += len(sent)
        # print(count)

        print(f"RESULTS FOR ALGORITHM 1 {package1}!!!!!!!!!!!!!!!!!!!!!!!")
        res1 = instance1.evaluate(X1, Y1)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # print("RESULTS LENGTH!!!!!!!!!!!!!!!!!")
        # print(len(res1))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("RESULT!!!!!!!!!!!!!!!!!!!!!1")
        # print(res1)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Assume instance 2 is CRF
        instance2 = class2_()

        # print("TOKENS LABELS LENGTH")
        # count = 0
        # for sent in tokens_labels:
        #     count += len(sent)
        # print(count)

        X2, Y2 = instance2.transform_sequences(tokens_labels)

        # print("LENGTH OF FATURIZED REPRESENTATION")
        # count = 0
        # for sent in X1:
        #     count += len(sent)
        # print(count)

        print(f"RESULTS FOR ALGORITHM 2 {package2}!!!!!!!!!!!!!!!!!!!!!!!")
        res2 = instance2.evaluate(X2, Y2)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        #print("CRF FEATURES DATA")
        #print(X2)
        # print("DATA THAT WE CAN TRY TO GET FROM CRF")
        # crf_dat = []
        # for lis in X2:
        #     for el in lis:
        #         dat = el["word.lower()"]
        #         crf_dat.append(dat)
        # print(crf_dat)
        # print(len(crf_dat))

        # print("RESULTS LENGTH!!!!!!!!!!!!!!!!!")
        # print(len(res2))
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print("RESULT!!!!!!!!!!!!!!!!!!!!!1")
        # print(res2)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        #print(f"LENGTH OF  GROUND TRUTH IS {len(ground_truth)}")

        assert len(res1) == len(res2), "if lengths of output sequences are not equal, then we cannot do evaluation of combination of algorithms, since what we are going to combine. Check evaluation() method of algorithms and make them agree on their output seuqences."
        # Cannot do this assertion due to different nature of algorithms
        #assert Y1 == Y2, "We should have the same ground truth labels for algorithms, as on that point we only transformed given sequences and not performed any NER"
        assert len(res1) == len(ground_tokens)

        result_labels = []
        # Now do union/intersection like it is done in mask_framework


        if resolution == "union":
            for i, j, k in zip(range(len(res1)), range(len(ground_tokens)), range(len(ground_truth))):
                alg1_pred = res1[i]
                alg2_pred = res2[i]

                union = ""
                if (alg1_pred == "O") and (alg2_pred != "O"):
                    print(f"Alg1 {algorithm1} recognised token {ground_tokens[j]} as 'O' when {algorithm2} recognised token as {alg2_pred}. True label is {ground_truth[k]}")
                    union = alg2_pred
                elif (alg2_pred == "O") and (alg1_pred != "O"):
                    print(f"Alg1 {algorithm1} recognised token {ground_tokens[j]} as {alg1_pred} when {algorithm2} recognised token as 'O'. True label is {ground_truth[k]}")
                    union = alg1_pred
                elif (alg1_pred != "O") and (alg2_pred != "O"):
                    if alg1_pred == alg2_pred:
                        #print(f"Both {algorithm1} and {algorithm2} recognised token {ground_tokens[j]} as {alg1_pred}. True label is {ground_truth[k]}")
                        union = alg1_pred
                    else: # it may happen, in that case take a token recognised with current entity_name we cater for (?)
                    # if labels are not equal -> choose label of leading algorithm 1
                        print(f"There is a debate: {algorithm1} recognised a token {ground_tokens[j]} as {alg1_pred} and {algorithm2} recognised it as {alg2_pred}. True label is {ground_truth[k]}")
                        print(f"Human analyst should look at this problem manually, for now this entity is set as {alg1_pred} as algorithm 1 is the lead")
                        union = alg1_pred

                else:
                    union = "O"

                result_labels.append(union)

        elif resolution == "intersection":
            for i, j in zip(range(len(res1)), range(len(ground_tokens))):
                alg1_pred = res1[i]
                alg2_pred = res2[i]

                intersection = ""
                if (alg1_pred == "O") or (alg2_pred == "O"):
                    intersection = "O"
                elif (alg1_pred != "O") and (alg2_pred != "O") and (alg1_pred == alg2_pred):
                    #print(f"Both {algorithm1} and {algorithm2} recognised token {ground_tokens[j]} as {alg1_pred}")
                    intersection = alg1_pred
                elif (alg1_pred != "O") and (alg2_pred != "O") and (alg1_pred != alg2_pred):
                    # if labels are not equal -> choose label of leading algorithm 1
                    print(f"{algorithm1} recognised a token {ground_tokens[j]} as {alg1_pred} and {algorithm2} recognised it as {alg2_pred}")
                    print(f"Human analyst should look at this problem manually, for now this entity is set as {alg1_pred} as algorithm 1 is the lead")
                    intersection = alg1_pred

                result_labels.append(intersection)

        # Somewhere we should have true labels array, need to evaluate wrt to it all the statistics.
        assert  len(result_labels) == len(ground_truth), f"if lengths of output sequences are not equal, then we cannot do evaluation of combination of algorithms, since what we are going to combine. Check evaluation() method of algorithms and make them agree on their output seuqences. Length of result labels is {len(result_labels)} and length of groun truth is {len(ground_truth)}"
        labels = ["ID", "PHI", "NAME", "CONTACT", "DATE", "AGE", "PROFESSION", "LOCATION"]
        print("COMBINATION OF ALGORITMHS STATISTICS REPORT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Validation Accuracy: {}".format(accuracy_score(ground_truth, result_labels)))  # was other way around, why?
        print("Validation F1-Score: {}".format(f1_score(ground_truth, result_labels, average='weighted'))) # correct
        print(classification_report(ground_truth, result_labels, digits=4, labels=labels))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


##########################################################################################################
# One algorithm case, just call evaluate() function
    else:
        tokens_labels = utils.spec_tokenizers.tokenize_to_seq(documents)
        ground_truth = []
        ground_tokens = []
        for sequence in tokens_labels:
            for t, l in sequence:
                #print("{}\t{}".format(t,l))
                ground_truth.append(l)
                ground_tokens.append(t)

        package = "ner_plugins."+ algorithm
        inpor = importlib.import_module(package)
        # find a class and instantiate
        class_ = getattr(inpor, algorithm)
        instance = class_()
        X,Y = instance.transform_sequences(tokens_labels)
        res = instance.evaluate(X,Y)

        #for r, t in zip(res, ground_tokens):
            #print(f"Algorithm {algorithm} recognised token {t} as {r}")

    print("Done!")
