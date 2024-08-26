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

"""*mask_framework.py* --
Main MASK Framework module
Code by: Nikola Milosevic
"""

from cmath import nan
from os import listdir, path, makedirs
from os.path import isfile, join
from yaml import full_load
import xml.etree.ElementTree as ET

import re
import datetime
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.tokenize.util import align_tokens
import re
from pandas import DataFrame, Series, NA, concat 
# To calculate execution time
import time
import pandas as pd
from utils.spec_tokenizers import tokenize_and_preserve_labels
# Apply the patched _process method to medcat's TransformersNER
from patches.medcat.ner.transformers_ner import PatchedTransformersNER
from medcat.ner.transformers_ner import TransformersNER
TransformersNER._process = PatchedTransformersNER._process
from mask_output import MaskOutput
from configuration import Configuration
# TODO: some algorithms like CRF and its variations do not like files to mask in .xml format, which causes pai for the evaluation on test set of their performance.
# I am not sure why this problem was not addressed earlier, but that is for the future (summer?) to solve.

# CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

# # Clean xml file from xml tags not to crash de-identification algorithms (except BERT, BERT is ok with it)
# def cleanxml(raw_xml_content):
#   cleantext = re.sub(CLEANR, '', raw_xml_content)
#   return cleantext

_treebank_word_tokenizer = TreebankWordTokenizer()

def consolidate_NER_results(final_sequences, text):
    """
    Function that from a list of sequences returned from the NER function is updated with spans
    :param final_sequences: Sequences returned from NER function. Sequence is a array of arrays of tokens in format (token,label).
    :param text: full text article
    :return: a list of tuples that includes spans in the following format: (token,label,span_begin,span_end)
    """
    
    fin = []
    idx = 0
    text_idx = 0
    fin_seqs = []
    for ss, sent in enumerate(final_sequences):
        tokens, labels, spans = [],[],[]
        l_ind = []
        for tok, lab in sent:
            tokens.append(tok)
            labels.append(lab)
            l_ind.append(len(tok))

        # to make tokens comparable with bert output
        tokens, labels = tokenize_and_preserve_labels(tokens, labels)
        for l, lab in enumerate(labels):
            if (tokens[l] == "/" or tokens[l] == "," or tokens[l] == "-" or tokens[l] == "\\" or tokens[l] == ".") and (lab == "DATE" or lab == "NAME" or lab == "LOCATION" or lab == "ID" or lab == "CONTACT" or lab == "PROFESSION"):
                labels[l] = 'O'

        # # optional output with spans included
        spans = align_tokens(tokens, text[text_idx:])
        start,end = zip(*spans)
        start, end = [text_idx + i for i in start] , [text_idx + j for j in end]
        sub_idx = list(range(idx, idx + len(tokens)))
        fin.extend(list(zip(tokens, labels, start, end))) # all tups (tok, lab ,start, end)
        fin_seqs.append(list(zip(sub_idx, tokens, labels, start, end))) # fin_seqs tups (tok, lab ,start, end) grouped by sent
        idx = idx + len(tokens)
        text_idx = text_idx + sum(l_ind)

    fin_output = []
    dic_str = {i.span()[0]:i for i in re.finditer('[Cc]annot',text)}
    dic_end = {i.span()[1]:i for i in re.finditer('[Cc]annot',text)}
    for tok, lab, start, end  in fin:
        # To manage the 'cannot' difference in tokenization between BERT and treebank tokenizers and make them comparable in overal_result
        if dic_str and start in dic_str.keys() and tok != dic_str[start].group():
            fin_output.append(tuple([dic_str[start].group(), lab, dic_str[start].span()[0], dic_str[start].span()[1]]))
            
        elif dic_str and end in dic_end.keys() and tok != dic_end[end].group():
            pass

        elif (tok == "/" or tok == "," or tok == "-" or tok == "\\" or tok == ".") and (lab == "DATE" or lab == "NAME" or lab == "LOCATION" or lab == "ID" or lab == "CONTACT" or lab == "PROFESSION") :
            fin_output.append(tuple([tok, "O", start, end]))
        else:
            fin_output.append(tuple([tok, lab, start, end]))
        
    return fin_output


def compare_results(resolution, alg_result_1, alg_result_2):
    # Now the fun part
    # We consider 3-4 cases:

    overall_result = []
    # loop over indices of arrays as we will need to look in thee future as well as in the past
    for alg1_idx, alg2_idx in zip(range(len(alg_result_1)), range(len(alg_result_2))): # alg1 and alg2 of shape: (token, label, span_min, span_max)
        # Assert that we compare elements with the same tokens
        assert alg_result_1[alg1_idx][0] == alg_result_2[alg2_idx][0], f"Tokens {alg_result_1[alg1_idx]} and {alg_result_2[alg2_idx]} are not equal"
        # If labels or spans are not equal, this is bad. You need to align outputs of models in the fashin of BERT.
        # To be discussed in the documentation.

        assert alg_result_1[alg1_idx][2] == alg_result_2[alg2_idx][2], f"Mismatch of span (lower bound RES1 : {alg_result_1[alg1_idx][2]} RES2 : {alg_result_2[alg2_idx][2]}) for a pair of comparable tokens!"
        assert alg_result_1[alg1_idx][3] == alg_result_2[alg2_idx][3], f"Mismatch of span (upper bound RES1 : {alg_result_1[alg1_idx][3]} RES2 : {alg_result_2[alg2_idx][3]}) for a pair of comparable tokens!"


        # TODO: solve alignment problems to the style of BERT alignment. Need to change al CRF algorithms and check fro Glove BiLSTM.
        # CASE 1: If for certain token algorithm1 returns "O" and algorithm2 returns "ENTITY_NAME", use "ENTITY_NAME" overal

        alg1_pred = alg_result_1[alg1_idx][1]
        alg2_pred = alg_result_2[alg2_idx][1]

        result = []

        if resolution == "union":
            union = ""
            if (alg1_pred == "O") and (alg2_pred != "O"):
                union = alg2_pred
            elif (alg2_pred == "O") and (alg1_pred != "O"):
                union = alg1_pred
            elif (alg1_pred != "O") and (alg2_pred != "O"):
                if alg1_pred == alg2_pred:
                    union = alg1_pred
                else:
                    union = alg1_pred + '/' + alg2_pred
                    # raise Exception(f"Problem of intersection of dstinct token labels has occured")
            else:
                union = "O"
            result.append(alg_result_1[alg1_idx][0])    # token
            result.append(union) # label
            result.append(alg_result_1[alg1_idx][2]) # lower bound of span
            result.append(alg_result_1[alg1_idx][3]) # upper bound of span
            result = tuple(result)

        elif resolution == "intersection":
            intersection = ""
            if (alg1_pred == "O") or (alg2_pred == "O"):
                intersection = "O" # union = "O"
            elif (alg1_pred != "O") and (alg2_pred != "O") and (alg1_pred == alg2_pred):
                intersection = alg1_pred
            elif (alg1_pred != "O") and (alg2_pred != "O") and (alg1_pred != alg2_pred):
                intersection = alg1_pred + '/' + alg2_pred  #Both algorithms labels

            result.append(alg_result_1[alg1_idx][0])    # token
            result.append(intersection) # label
            result.append(alg_result_1[alg1_idx][2]) # lower bound of span
            result.append(alg_result_1[alg1_idx][3]) # upper bound of span
            result = tuple(result)
        
        if len(result) != 0:
            overall_result.append(result)
    return overall_result

def eHost_output_generator(tokens_tbl, output_path):
    tbl = tokens_tbl.copy()
    if sum(tbl['label_final'].str.find('/') != -1) > 0:

        tbl['label_final'] = tbl['label_final'].str.split('/')
        tbl = tbl.explode('label_final').reset_index(drop=True)

        #remove duplicate rows for same token and sort by token_start
        tbl = tbl.drop_duplicates(['file','token','token_start', 'token_end'], keep = 'first')\
                               .sort_values(by = ['file','token_start']) \
                               .reset_index(drop = True)

    for file in tbl['file'].unique():

        for alg in tbl['alg_name'].unique():
            src_file = ET.Element("annotations", textSource = file)
            tree = ET.ElementTree(src_file)

            for idx, row in enumerate(tbl[(tbl['file'] == file) & (tbl['alg_name'] == alg)].index):

                tok_id = tbl['alg_name'][row] + "_Instance_" + str(idx + 1)
                annotation = ET.SubElement(src_file, "annotation")
                label = ET.SubElement(src_file, "classMention", id = tok_id)

                ET.SubElement(annotation, "mention", id = tok_id,)
                ET.SubElement(annotation, "annotator", id = tbl['alg_name'][row]).text = tbl['alg_name'][row]
                ET.SubElement(annotation, "span", start=str(tbl['token_start'][row]), end = str(tbl['token_end'][row]))
                ET.SubElement(annotation, "spannedText").text = tbl['token'][row]
                ET.SubElement(annotation, 'creationDate').text = tbl['run_date'][row]
                ET.SubElement(label, "mentionClass", id = tbl['label_final'][row]).text = tbl['token'][row]

            end_notes = ET.SubElement(src_file,"eHOST_Adjudication_Status", version = "1.0")
            ET.SubElement(end_notes, "Adjudication_Selected_Annotators", version="1.0")
            ET.SubElement(end_notes, "Adjudication_Selected_Classes", version="1.0")
            classes = ET.SubElement(end_notes, "Adjudication_Others")
            ET.SubElement(classes, "CHECK_OVERLAPPED_SPANS").text = 'false'
            ET.SubElement(classes,"CHECK_ATTRIBUTES").text = 'false'
            ET.SubElement(classes, "CHECK_RELATIONSHIP").text = 'false'
            ET.SubElement(classes, "CHECK_CLASS").text = 'false'
            ET.SubElement(classes, "CHECK_COMMENT").text = 'false'

            ET.indent(src_file) #  this line only runs if python version 3.9+ (is key to have it otherwise xml files won't open on ehost)

            if not path.exists(output_path + '/' + alg + '/'):
                makedirs(output_path + '/' + alg + '/', exist_ok = True)

            file_nm = output_path + '/' + alg + '/' + file + '.knowtator.xml'
            myfile = open(file_nm, "wb")
            myfile.write(ET.tostring(src_file, encoding='UTF-8',method='xml',xml_declaration = True))
            myfile.close()
    return print(f"eHost files have been successfully created")


def main():
    """Main MASK Framework function
               """
    print("Welcome to MASK")
    cf = Configuration()
    cf.load()
    cf.instantiate()
    log_file = open('log_mask_running.log', 'w', encoding='utf-8')
    mask_output = MaskOutput(cf.project_name, log_file)
    execute(cf, mask_output)
    log_file.close()

def execute(cf, mask_output):
    data = [f for f in listdir(cf.dataset_location) if isfile(join(cf.dataset_location, f))]
    mask_output.input_max = len(data)
    mask_output.algorithm_max = len(cf.algorithms)
    mask_output.begin()

    tokens_found = DataFrame()
    elements = Series()
    times = []

    # Ensure full output directories exist
    if not path.exists(cf.data_output):
        makedirs(cf.data_output, exist_ok=True)
    if not path.exists(cf.csv_output):
        makedirs(cf.csv_output, exist_ok=True)

    for file_index, file in enumerate(data):
        print(f'********** {file} **********\n\n')
        mask_output.set_current_input(file_index + 1)
        mask_output.set_current_algorithm(0)
        input_path = cf.dataset_location + "/" + file
        output_path = cf.data_output + "/" + file
        csv_path = cf.csv_output + '/removed_tokens.csv'
        masked_file_log = mask_output.mask(input_path, output_path, csv_path)
        masked_file_log.begin()

        text = open(input_path, 'r').read()
        new_text = text   # text is an original text
        tokens2rm = DataFrame()

        time_all_1 = time.time()
        for i in range(0, len(cf.algorithms)): # for each function call, algorithms - list of disctionaries
            alg = cf.algorithms[i]
            next_alg = {}
            next_alg_entity_name = ""
            masking_type = alg['masking_type']
            current_alg_entity_name = alg["entity_name"]

            if i != (len(cf.algorithms) - 1): # last instruction in the algorithms dictionary does not have any future
                next_alg = cf.algorithms[i+1] # for the check if next algorithm's entity is the same as the current one
                next_alg_entity_name = next_alg["entity_name"]

            # if this is the case, we know that we will need to compare the results of 2 algorithms outputs.
            if current_alg_entity_name == next_alg_entity_name:
                continue # as we will combine the output for entity on the next run to run masking only once instead of twice!
            else:
                previous_alg = cf.algorithms[i-1]
                previous_alg_entity_name = previous_alg["entity_name"]

                # if previous_alg_entity_name == current_alg_entity_name:
                perform_resolution = (previous_alg_entity_name == current_alg_entity_name) and 'resolution' in alg
                if perform_resolution: #previous_alg['algorithm'] != alg['algorithm']:
                    alg_result_1 = cf.instances[previous_alg['algorithm']].perform_NER(new_text)
                    alg_result_1 = consolidate_NER_results(alg_result_1, new_text) # (token, label, token_start, token_end)

                    alg_result_2 = cf.instances[alg['algorithm']].perform_NER(new_text)
                    alg_result_2 = consolidate_NER_results(alg_result_2, new_text) # (token, label, token_start, token_end)

                    # Do function compare_results(result1, result2) that returns overall result
                    overal_result = compare_results(alg['resolution'], alg_result_1, alg_result_2)

                else:
                    alg_result_1 = cf.instances[alg['algorithm']].perform_NER(new_text)
                    alg_result_1 = consolidate_NER_results(alg_result_1, new_text) # (token, label, span_min, token_end)
                    overal_result = alg_result_1
                
                file_df = DataFrame(overal_result, columns = ['token', 'label', 'token_start', 'token_end'])

                if perform_resolution:
                    file_df = file_df.merge(DataFrame(alg_result_1, columns = file_df.columns)['label'] , suffixes = ['_final', None], left_index = True, right_index = True)\
                           .merge(DataFrame(alg_result_2, columns = file_df.columns)['label'], suffixes = ['_alg1', '_alg2'], left_index = True, right_index = True)
                    file_df = file_df[((file_df.label_alg1 == current_alg_entity_name) | (file_df.label_alg2 == current_alg_entity_name))]

                else:
                    file_df = file_df.merge(DataFrame(alg_result_1, columns = file_df.columns)['label'] , suffixes = ['_final', '_alg1'], left_index = True, right_index = True)
                    file_df = file_df[file_df.label_final == current_alg_entity_name]

                file_df['file'] = file
                file_df['token_size'] = [len(file_df.loc[idx,'token']) for idx in file_df.index] # calculates size of tokens to be masked/redacted
                file_df['masking_type'] = masking_type # appends masking_type
                file_df['new_token'] = [cf.instances[alg['masking_class']].mask(word) if masking_type == 'Mask' else 'XXX' for word in file_df['token']] # masks/redacts tokens
                file_df['new_token_size'] = [len(file_df.loc[idx,'new_token']) for idx in file_df.index] # calculates size of masked/redacted tokens
                file_df['shift'] = file_df['new_token_size'] - file_df['token_size']
                # file_df['alg_name'] is the alg name used to create eHost files, When union/intersection name is given
                # by the union/intersection of alphabetically ordered alg1, alg2
                file_df['alg_name'] = f"_{alg['resolution'].upper()}_".join(sorted({previous_alg['algorithm'], alg['algorithm']}))  if previous_alg_entity_name == current_alg_entity_name and 'resolution' in alg else alg['algorithm']
                file_df['run_date'] = datetime.datetime.now().astimezone().strftime("%d %b %Y %H:%M:%S %Z")
                
                for r in file_df.itertuples(index=False):
                    masked_file_log.replace(r.alg_name, r.label_final, masking_type, r.token, r.new_token,
                                            r.token_start, r.token_end)

            tokens2rm = concat([tokens2rm,file_df]).reset_index(drop = True)
            tokens_found = concat([tokens_found,file_df]).drop(['shift', 'token_size','new_token_size'], axis = 1).reset_index(drop = True)
        print(f"\n\nALGS: {tokens_found.alg_name.unique()}\n\n")

        tokens2rm['label_final'] = tokens2rm['label_final'].str.split('/')
        tokens2rm = tokens2rm.explode('label_final').reset_index(drop=True)

        #remove duplicate rows for same token and sort by token_start
        tokens2rm = tokens2rm.drop_duplicates(['file','token','token_start', 'token_end'], keep = 'first')\
                                       .sort_values(by = ['file','token_start']) \
                                       .reset_index(drop = True)
        # recalculate token_start for replacement
        for i in tokens2rm.index:
            tokens2rm.loc[i,'token_start'] = tokens2rm.loc[i,'token_start'] + sum(tokens2rm.loc[0:i-1,'shift'])

        tokens2rm['token_size'] = tokens2rm['token_size'].astype('int64')
        tokens2rm['token_start'] = tokens2rm['token_start'].astype('int64')
        tokens2rm['new_token_size'] = tokens2rm['new_token_size'].astype('int64')
        tokens2rm['token_end'] = tokens2rm['token_start'] + tokens2rm['token_size']

        for i in tokens2rm.index:
            # replace tokens in text with new masked/redacted tokens
            new_text = new_text[:tokens2rm.loc[i,'token_start']] + tokens2rm.loc[i,'new_token'] + new_text[tokens2rm.loc[i, 'token_end'] :]
            #records each replacement in log
            mask_output.log(f"{tokens2rm.loc[i,'masking_type'] + 'ed'} token : {tokens2rm.loc[i,'token']} --> New token : {tokens2rm.loc[i,'new_token']} \n")


        time_all_2 = time.time()
        times.append((time_all_2 - time_all_1))

        print(f"TIME to execute FULL NER FOR {file} was {time_all_2 - time_all_1} \n\n")

        # write into output files
        file_handler = open(output_path, "w")
        file_handler.write(new_text)
        file_handler.close()


        cnt = tokens2rm.groupby(['label_final']).count()['token'].rename('Count')
        elements = concat([elements,cnt])
        # logs total masked/redacted for each entity in file
        mask_output.log('\n' + '\n'.join([' '.join(['Elements detected as',x, 'in', file, ' : ', str(y)]) for x, y in zip(cnt.index,cnt)]) + '\n')

        masked_file_log.finish()

    elements = elements.groupby(axis = 0, level = 0).sum()

    mask_output.log(f'\n\n ========================== FINAL STATS ======================== \n\n')
    # print(f'\n\n TOTAL  tokens found: {tokens_found.shape[0]}\n\n')
    if tokens_found.shape[0] > 0:
        tokens_found[['alg_name','file','token','label_final', 'token_start', 'token_end']].to_csv(csv_path, index = False)

        if sum(tokens_found['label_final'].str.find('/') != -1) > 0:
            mask_output.log(f"\n WARNING: {tokens_found[tokens_found['label_final'].str.find('/') != -1].shape[0]} token(s) masked/redacted with discrepancies detected in resolution.\n\t")

        mask_output.log('\n' + '\n'.join([' '.join(['Total elements masked/redacted as', x, ' : ', str(y)]) for x, y in zip(elements.index,elements)]) + '\n')

        if cf.ehost_output:
            # creates and saves ehost outputs
            eHost_output_generator(tokens2rm, cf.ehost_output)


    avg = Series(times).mean().__round__(2)
    std = Series(times).std().__round__(2)
    print(f"AVERAGE TIME TO EXECUTE ON ALL NER ENTITIES FOR ALL INPUT DOCUMENTS WAS {round(avg,2)} Secs")
    print(f"STD OF TIME TO EXECUTE ON ALL NER ENTITIES FOR ALL INPUT DOCUMENTS WAS {round(std,2)} Secs")

    return mask_output


if __name__=="__main__":
    main()
