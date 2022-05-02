# coding: utf-8

import torch
import json
import random
import traceback
import torch.nn.functional as F
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, SoftTemplate, MixedTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from typing import Dict, List, Any
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


class PromptEntityVector(object):
    """
    利用promot soft模型生成对应的mention和entity向量
    """
    def __init__(self,
                 classes: List[str],
                 entity_data_path: str,
                 mention_data_path: str,
                 label_words: Dict[str, List],
                 use_cuda: bool = True,
                 plm_name: str = "bert",
                 plm_path: str = "bert-base-cased",
                 template_text: str = '{"meta": "mention_context"} {"meta": "mention_name"} and the '
                                      '{"meta":"cui_type"} {"meta":"cui_name"} have the same meaning, '
                                      'is it correct? {"mask"}.',
                 load_model_path: str = "",
                 template_name: str="manual",
                 verbalizer_name: str="manual",
                 batch_size: int=10):

        self.classes = classes
        self.entity_data_path = entity_data_path
        self.mention_data_path = mention_data_path
        self.label_words = label_words
        self.use_cuda = use_cuda
        self.plm_name = plm_name
        self.plm_path = plm_path
        self.template_text = template_text
        self.label_id_map = {"positive": 1, "negative": 0}
        self.load_model_path = load_model_path
        self.template_name = template_name
        self.verbalizer_name = verbalizer_name
        self.batch_size = batch_size
        self.golden_type_file = "../data/MRSTY.RRF"
        self.type_file = "../data/SRDEF"
        self.load_data()
        self.load_model()

    def load_data(self):
        self.type_id_name = dict()
        self.golden_id_type = dict()

        with open(self.golden_type_file) as f:
            for line in f:
                ent_id, type_id, _, type_name = line.strip("|").split("|")[0: 4]
                if ent_id not in self.golden_id_type:
                    self.golden_id_type[ent_id] = [type_id]
                else:
                    self.golden_id_type[ent_id].append(type_id)

        with open(self.type_file) as f:
            for line in f:
                _, type_id, type_name, _, type_desc = line.strip().split("|")[0: 5]
                if type_id not in self.type_id_name:
                    self.type_id_name[type_id] = {"type_name": type_name, "type_desc": type_desc, "type_id": type_id}
                else:
                    raise RuntimeError("repeat in type_id_name !!")
        self.type_id_name["UnknownType"] = {"type_name": "unknown type", "type_desc": "unknown type",
                                            "type_id": type_id}

    def load_model(self):
        # load plm
        self.plm, self.tokenizer, model_config, self.WrapperClass = load_plm(self.plm_name, self.plm_path)

        # define template
        if self.template_name == "manual":
            self.promptTemplate = ManualTemplate(text=self.template_text, tokenizer=self.tokenizer)
        elif self.template_name == "soft":
            self.promptTemplate = SoftTemplate(model=self.plm, text=self.template_text, tokenizer=self.tokenizer)
        elif self.template_name == "mix":
            self.promptTemplate = MixedTemplate(model=self.plm, text=self.template_text, tokenizer=self.tokenizer)
        else:
            raise RuntimeError("template name is not right !")

        # define verbalizer
        if self.verbalizer_name == "manual":
            self.promptVerbalizer = ManualVerbalizer(classes=self.classes, label_words=self.label_words, tokenizer=self.tokenizer)
        elif self.verbalizer_name == "soft":
            self.promptVerbalizer = SoftVerbalizer(plm=self.plm, classes=self.classes, label_words=self.label_words, tokenizer=self.tokenizer)
        else:
            raise RuntimeError("verbalizer name is not right !")

        # define prompt model
        self.prompt_model = PromptForClassification(template=self.promptTemplate, plm=self.plm, verbalizer=self.promptVerbalizer)
        if self.load_model_path:
            state_dict = torch.load(self.load_model_path)
            self.prompt_model.load_state_dict(state_dict)
        self.prompt_model.eval()
        if self.use_cuda:
            self.prompt_model = self.prompt_model.cuda()
        self.prompt_model = self.prompt_model.prompt_model
        print("load old model succeeded...")

    def load_ent_dataset(self, inputs, types):
        dataset = []
        tokenize_len = []
        for index, item in enumerate(zip(inputs, types)):
            guid = index
            meta = {}
            meta["mention_context"] = item[1]["type_desc"]
            meta["mention_name"] = item[0]
            meta["cui_type"] = item[1]["type_name"]
            meta["cui_name"] = item[0]
            dataset.append(InputExample(guid=guid, meta=meta))
            tokens_lm = self.tokenizer.tokenize(meta["cui_name"])
            tokenize_len.append(len(tokens_lm))
            
        return dataset, torch.tensor(tokenize_len)

    def load_mention_dataset(self, mention_names, mention_contexts):
        dataset = []
        tokenize_len = []
        for index, item in enumerate(zip(mention_names, mention_contexts)):
            guid = index
            meta = {}
            meta["mention_context"] = item[1]
            meta["mention_name"] = item[0]
            meta["cui_type"] = "unknown type"
            meta["cui_name"] = item[0]
            dataset.append(InputExample(guid=guid, meta=meta))
            tokens_lm = self.tokenizer.tokenize(meta["mention_name"])
            tokenize_len.append(len(tokens_lm))
        return dataset, torch.tensor(tokenize_len)

    def process_batch(self, inputs: List[Any], tokenizer, promptTemplate, WrapperClass):
        prompt_loader = PromptDataLoader(dataset=inputs, tokenizer=tokenizer,
                                         template=promptTemplate, tokenizer_wrapper_class=WrapperClass,
                                         max_seq_length=256, batch_size=len(inputs))
        return prompt_loader.__iter__().__next__()

    def gener_entity_vector(self, batch_cui_id, batch_cui_name, batch_cui_type, wo=None):
        dataset, tokenize_len = self.load_ent_dataset(batch_cui_name, batch_cui_type)
        try:
            inputs = self.process_batch(dataset, self.tokenizer, self.promptTemplate, self.WrapperClass)
        except Exception as e:
            res = {"batch_cui_id": batch_cui_id, "batch_cui_name": batch_cui_name, "batch_cui_type": batch_cui_type}
            with open("gener_entity_vector.error", "a+") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            return

        if self.use_cuda:
            inputs = inputs.cuda()
        outputs = (self.prompt_model(inputs)).hidden_states[0]
        soft_index = torch.where(inputs['soft_token_ids'] == 2)
        new_outputs = []
        for ent_num, sort_index in enumerate(soft_index[1]):
            begin_index = sort_index - tokenize_len[ent_num]
            end_index = sort_index
            ent_vector = torch.mean(outputs[ent_num, begin_index: end_index, :], dim=0)
            new_outputs.append(ent_vector.unsqueeze(0))
        outputs = torch.cat(new_outputs, dim=0)
            
        #soft_index = (soft_index[0], soft_index[1]-1)
        #outputs = outputs[soft_index]
        # N * num_soft_tokens * 768
        #outputs = outputs.view(inputs['soft_token_ids'].shape[0], -1, outputs.shape[1])
        # entity rep
        #outputs = outputs[:, 1, :]
        for cur_cui_id, cur_cui_name, output in zip(batch_cui_id, batch_cui_name, outputs):
            res = {"cui_id": cur_cui_id, "cui_name": cur_cui_name, "vector": output.cpu().tolist()}
            if wo:
                wo.write(json.dumps(res, ensure_ascii=False) + "\n")

    def gener_mention_vector(self, batch_mention_name, batch_mention_context, batch_golden_cui, wo=None):
        dataset, tokenize_len = self.load_mention_dataset(batch_mention_name, batch_mention_context)
        try:
            inputs = self.process_batch(dataset, self.tokenizer, self.promptTemplate, self.WrapperClass)
        except Exception as e:
            res = {"batch_mention_name": batch_mention_name, "batch_mention_context": batch_mention_context, "batch_golden_cui": batch_golden_cui}
            with open("gener_mention_vector.error", "a+") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            return

        if self.use_cuda:
            inputs = inputs.cuda()
        outputs = (self.prompt_model(inputs)).hidden_states[0]
        soft_index = torch.where(inputs['soft_token_ids'] == 1)
        new_outputs = []
        for ent_num, sort_index in enumerate(soft_index[1]):
            begin_index = sort_index - tokenize_len[ent_num]
            end_index = sort_index
            ent_vector = torch.mean(outputs[ent_num, begin_index: end_index, :], dim=0)
            new_outputs.append(ent_vector.unsqueeze(0))
        outputs = torch.cat(new_outputs, dim=0)
        #soft_index = (soft_index[0], soft_index[1]-1)
        #outputs = outputs[soft_index]
        # N * num_soft_tokens * 768
        #outputs = outputs.view(inputs['soft_token_ids'].shape[0], -1, outputs.shape[1])
        # mention rep
        #outputs = outputs[:, 0, :]
        for cur_mention_name, cur_mention_context, cur_golden_cui, output in zip(batch_mention_name, batch_mention_context, batch_golden_cui, outputs):
            res = {"mention_name": cur_mention_name, "mention_context": cur_mention_context, "golden_cui": cur_golden_cui, "vector": output.cpu().tolist()}
            if wo:
                wo.write(json.dumps(res, ensure_ascii=False) + "\n")

    def predict_entity(self):
        wo = open("soft_template_entity_vector.pubmed", "w")
        batch_cui_name = list()
        batch_cui_id = list()
        batch_cui_type = list()
        with open(self.entity_data_path) as f:
            for line in f:
                ent_id, ent_name = line.strip().split("||")
                ent_id = ent_id.strip()
                ent_name = ent_name.strip()
                if len(batch_cui_name) < self.batch_size:
                    batch_cui_name.append(ent_name)
                    batch_cui_id.append(ent_id)
                    batch_cui_type.append(self.type_id_name[self.golden_id_type[ent_id][0]])
                else:
                    self.gener_entity_vector(batch_cui_id, batch_cui_name, batch_cui_type, wo)
                    batch_cui_name = []
                    batch_cui_id = []
                    batch_cui_type = []
                    batch_cui_name.append(ent_name)
                    batch_cui_id.append(ent_id)
                    batch_cui_type.append(self.type_id_name[self.golden_id_type[ent_id][0]])
            if len(batch_cui_name) > 0:
                self.gener_entity_vector(batch_cui_id, batch_cui_name, batch_cui_type, wo)
        wo.close()

    def predict_mention(self):
        wo = open("../data/plm_vector_compared/soft_template_mention_vector.prompt", "w")
        batch_mention_name = list()
        batch_mention_context = list()
        batch_golden_cui = list()
        with open(self.mention_data_path) as f:
            for line in f:
                json_data = json.loads(line.strip())
                golden_cui = json_data["golden_cui"]
                smaple = json_data["sample_cadidates"][0]
                mention_name = smaple["mention_name"]
                mention_context = smaple["mention_context"]
                if len(batch_mention_name) < self.batch_size:
                    batch_mention_name.append(mention_name)
                    batch_mention_context.append(mention_context)
                    batch_golden_cui.append(golden_cui)
                else:
                    self.gener_mention_vector(batch_mention_name, batch_mention_context, batch_golden_cui, wo)
                    batch_mention_name = []
                    batch_mention_context = []
                    batch_golden_cui = []
                    batch_mention_name.append(mention_name)
                    batch_mention_context.append(mention_context)
                    batch_golden_cui.append(golden_cui)

            if len(batch_mention_name) > 0:
                self.gener_mention_vector(batch_mention_name, batch_mention_context, batch_golden_cui, wo)
        wo.close()

    def test_error(self):
        wo = open("buchong_vector_2", "w")
        batch_cui_name = list()
        batch_cui_id = list()
        batch_cui_type = list()
        with open("buchong_2.out") as f:
            for line in f:
                json_data = json.loads(line.strip())
                ent_id = json_data["cui_id"]
                ent_name = json_data["cui_name"]
                if len(batch_cui_name) < self.batch_size:
                    batch_cui_name.append(ent_name)
                    batch_cui_id.append(ent_id)
                    batch_cui_type.append(self.type_id_name[self.golden_id_type[ent_id][0]])
                else:
                    self.gener_entity_vector(batch_cui_id, batch_cui_name, batch_cui_type, wo)
                    batch_cui_name = []
                    batch_cui_id = []
                    batch_cui_type = []
                    batch_cui_name.append(ent_name)
                    batch_cui_id.append(ent_id)
                    batch_cui_type.append(self.type_id_name[self.golden_id_type[ent_id][0]])

            if len(batch_cui_name) > 0:
                self.gener_entity_vector(batch_cui_id, batch_cui_name, batch_cui_type, wo)
        wo.close()


if __name__ == "__main__":
    # mix template predict
    pmt = PromptEntityVector(classes=["negative", "positive"],
                             entity_data_path="../data/umls2017aa_reference_ont.txt",
                             mention_data_path="../data/prompt_for_test",
                             label_words = {"negative": ["no"], "positive": ["yes"]},
                             use_cuda = True,
                             plm_name="bert",
                             plm_path="../pretrain/pubmed_bert",
                             template_text='{"meta": "mention_context"} {"meta": "mention_name", "shortenable": True} {"soft"} and the '
                                           '{"meta":"cui_type"} {"meta":"cui_name", "shortenable": True} {"soft"} have the same meaning, '
                                           'is it correct? {"mask"} {"soft"}.',
                             load_model_path="../old_model_saved/mix_template_model/best_12.th",
                             template_name="mix",
                             verbalizer_name="manual")
    #pmt.predict_entity()
    pmt.predict_mention()
