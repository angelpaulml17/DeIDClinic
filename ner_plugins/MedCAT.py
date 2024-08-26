import shutil
import sklearn_crfsuite
import pickle
from ner_plugins.NER_abstract import NER_abstract
from seqeval.metrics import accuracy_score
from sklearn.metrics import f1_score
import tarfile
import os.path
from medcat.cat import CAT
from medcat.utils.ner import make_or_update_cdb, deid_text
from medcat.ner.transformers_ner import TransformersNER
from medcat.config_transformers_ner import ConfigTransformersNER
import xml.etree.ElementTree as ET
import tarfile
import json
from medcat.cdb import CDB
from medcat.config import Config
from medcat.vocab import Vocab
from nltk.tokenize.util import align_tokens
from utils.spec_tokenizers import tokenize_fa, custom_span_tokenize, custom_word_tokenize, tokenize_and_preserve_labels
from nltk import sent_tokenize
from pandas import DataFrame

# class MedCAT(NER_abstract):
class MedCAT(NER_abstract):
    def __init__(self):
        print("PERFORM MedCAT init")
        model_pack_name="Models\deid_medcat_n2c2_modelpack.zip"
        self.cat = CAT.load_model_pack(model_pack_name)

    def perform_NER(self,text):
        # print('cat(text):', cat(text), '\n\n')
        # print("ANNOTATED TEXT")
        # anon_text2 = deid_text(cat, text, redact=True)
        # print(anon_text2)
        result = self.cat.get_entities(text)

        if len(result['entities'].values())!=0:

            new_tokens, new_labels,new_start,new_end = [], [],[], []

            for dic in result['entities'].values():
                tkn = custom_word_tokenize(dic['source_value']) # token splited in equivalent way to the other algs
                n = len(tkn)
                lbl = dic['cui'] # label detected by the model

                if lbl in ['DATE', 'DATE OF BIRTH']:
                    lbl = ['DATE']
                
                elif lbl in ['DOCTOR','PATIENT','USERNAME']:
                    lbl = ['NAME']

                elif lbl in ['STREET','STATE','CITY','ZIP','COUNTRY','LOCATION-OTHER','HOSPITAL']:
                    lbl = ['LOCATION']

                elif lbl in ['HOME','EMAIL','FAX', 'PHONE', 'TELEPHONE NUMBER']:
                    lbl = ['CONTACT']
                
                elif lbl in ['IDNUM', 'BIOID']:
                    lbl = ['ID']

                else:
                    lbl = [lbl]
                toks, labs = tokenize_and_preserve_labels(tkn, lbl * n)
                new_tokens.extend(toks)
                new_labels.extend(labs)
                new_start.extend(list(map(lambda x: x[0] + dic['start'], custom_span_tokenize(dic['source_value'],use_bert_tok=True))))
                new_end.extend(list(map(lambda x: x[1] + dic['start'], custom_span_tokenize(dic['source_value'],use_bert_tok=True))))

            new_result = list(zip(new_tokens,new_labels,new_start,new_end))
            
            final_sequences = tokenize_fa([text], use_bert_tok=True)
            spans = custom_span_tokenize(text, use_bert_tok= True, incl_tok_sent=True)

            i = 0
            for seq, seq_span in zip(final_sequences,spans):
                ori_seq = [tuple([tok,start,end]) for (tok,lab),(start,end) in zip(seq, seq_span)]
                for t,l,s,e in new_result:
                    if tuple([t,s,e]) in ori_seq:
                        idx = ori_seq.index(tuple([t,s,e]))
                        final_sequences[i][idx] = tuple([t,l])
                i = i + 1
        else:
            final_sequences = custom_word_tokenize(text, incl_tok_sent = True) 
            final_sequences = [list(zip(seq, ['O'] * len(seq))) for seq in final_sequences]

        return final_sequences
        


    #def transform_sequences(self,tokens_labels):
    #    print("TRANSFORM")

    #def learn(self, X_train,Y_train):
    #    print("TRAIN")
        
    def train(self,file_name,model_pack_path):
        print("TRAIN")
        TRAIN_DATA_PATH=file_name
        
        #TRAIN_DATA_PATH="deid_train_data.json"
        cdb = 'Models/deid_medcat_n2c2_modelpack/cdb.dat'
        cdb = CDB.load(cdb)
    
        cdb = make_or_update_cdb(TRAIN_DATA_PATH, cdb = cdb, min_count = 0)
        config = ConfigTransformersNER()
        config.general['test_size'] = 0.1 # Usually set this to 0.1-0.2
        config.general['model_name'] = 'bert-base-uncased'#can change using any BERT models from the hugging face hub. Judt copy the name of the model and put it in here.
        config.general
        ner = TransformersNER(cdb = cdb, config = config)
        ner.training_arguments.num_train_epochs = 1 # Use 5-10 normally
        # As we are running the training on a GPU that can [probably] handle a larger BS, we'll set it to 8
        ner.training_arguments.per_device_train_batch_size = 8
        ner.training_arguments.gradient_accumulation_steps = 1 # No need for acc
        ner.training_arguments.per_device_eval_batch_size = 8
        # For the metric to be used for best model we pick Recall here, as for deid that is most important
        ner.training_arguments.metric_for_best_model = 'eval_recall'

        df, examples, dataset = ner.train(TRAIN_DATA_PATH)
        #print(df)
        ####Editing the config file######
        cat = CAT(cdb=ner.cdb, addl_ner = ner)
        cat.config.version['description'] = "Model trained Deid: " + ', '.join(list(df.name.values))
        cat.config.version['location'] = "Publicaly available from the MedCAT repository: https://github.com/CogStack/MedCAT"
        cat.config.version['ontology'] = "Deid"
        ########CREATE MODEL########
        
        model_pack_name = cat.create_model_pack("Models", model_pack_name='Deid')
#       ######TEST PHASE##########
        #model_pack_name="/content/drive/MyDrive/Deid_Roberta.zip"
    
        return(model_pack_name)
    
    def evaluate(self, X_test,Y_test):
        print("EVALUATE")
        
    def to_json(self,root, file_name):
        #content = file_name.read()
        #root = ET.fromstring(content)
        text = root.find('TEXT').text
        tags = [(t.tag, t.attrib) for t in list(root.find('TAGS'))]
        annotations = []
        for tag in tags:
            annotations.append({
                'start': tag[1]['start'],
                'end': tag[1]['end'],
                'cui': tag[1]['TYPE'],
                'value': tag[1]['text']})
        doc = {'text': text, 'name': file_name, 'annotations': annotations}

        return doc
    def to_json_tar(self,filename_tar):
        tars=[tarfile.open(filename_tar,"r:gz")]
        data = {'projects': []}
        for tar in tars:
            p = {'name': tar.name, 'documents': []}
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read()
                    root = ET.fromstring(content)
                    doc = self.to_json(root, os.path.basename(member.name))
                    p['documents'].append(doc)
        data['projects'].append(p)
        # Save
        TRAIN_DATA_PATH="trainset.json"
        json.dump(data, open(TRAIN_DATA_PATH, 'w'))
        return(TRAIN_DATA_PATH)
   
    def make_tarfile(self,source_dir):
        compressed_file = shutil.make_archive(
        base_name='train_xml',   # archive file name w/o extension
        format='gztar',        # available formats: zip, gztar, bztar, xztar, tar
        root_dir=source_dir    # directory to compress
        )
        return (compressed_file)
