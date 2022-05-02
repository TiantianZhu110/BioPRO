# coding: utf-8

import torch
import json
import random
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


class PromptMedicinePredict(object):
    """
    医疗实体链接prompt预测
    """
    def __init__(self,
                 classes: List[str],
                 test_data_path: str,
                 label_words: Dict[str, List],
                 use_cuda: bool = True,
                 plm_name: str = "bert",
                 plm_path: str = "bert-base-cased",
                 template_text: str = '{"meta": "mention_context"} {"meta": "mention_name"} and the '
                                      '{"meta":"cui_type"} {"meta":"cui_name"} have the same meaning, '
                                      'is it correct? {"mask"}.',
                 load_model_path: str = "",
                 template_name: str="manual",
                 verbalizer_name: str="manual"):

        self.classes = classes
        self.test_data_path = test_data_path
        self.label_words = label_words
        self.use_cuda = use_cuda
        self.plm_name = plm_name
        self.plm_path = plm_path
        self.template_text = template_text
        self.label_id_map = {"positive": 1, "negative": 0}
        self.load_model_path = load_model_path
        self.template_name = template_name
        self.verbalizer_name = verbalizer_name
        self.load_model()

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
        print("load old model succeeded...")

    def load_dataset(self, inputs):
        dataset = []
        for index, json_item in enumerate(inputs):
            guid = index
            meta = {}
            meta["mention_context"] = json_item["mention_context"]
            meta["mention_name"] = json_item["mention_name"]
            meta["cui_type"] = json_item["cui_type"]
            meta["cui_name"] = json_item["cui_name"]
            dataset.append(InputExample(guid=guid, meta=meta))
        return dataset

    def process_batch(self, inputs: List[Any], tokenizer, promptTemplate, WrapperClass):
        prompt_loader = PromptDataLoader(dataset=inputs, tokenizer=tokenizer,
                                         template=promptTemplate, tokenizer_wrapper_class=WrapperClass,
                                         max_seq_length=256, batch_size=len(inputs))
        res = prompt_loader.__iter__().__next__()
        return res

    def predict(self):
        w_out = open("../data/top100/medmention_soft_predict_result.100", "w") 
        acc = 0
        num = 0
        with open(self.test_data_path) as f:
            for line in f:
                json_data = json.loads(line.strip())
                golden_cui = json_data["golden_cui"]
                dataset = self.load_dataset(json_data["sample_cadidates"])
                final_logits = []
                start_index = 0
                end_index = 10
                flag_continue = False

                while start_index < len(dataset):
                    try:
                        inputs = self.process_batch(dataset[start_index: end_index], self.tokenizer, self.promptTemplate, self.WrapperClass)
                    except Exception as e:
                        print("len limit is 256!!!")
                        flag_continue = True
                        break
                    if self.use_cuda:
                        inputs = inputs.cuda()
                    logits = self.prompt_model(inputs)
                    final_logits.append(logits.data)
                    start_index += 10
                    end_index += 10

                if flag_continue:
                    continue

                final_logits = torch.cat(final_logits, dim=0)
                if len(final_logits) != len(dataset):
                    raise RuntimeError("len(final_logits) != len(dataset) !!!")

                logits = F.softmax(final_logits, dim=-1)
                pred_index = torch.argmax(logits, dim=0).cpu().tolist()[1]
                json_data["predict_score"] = logits.cpu().tolist()
                w_out.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                pred_cui = json_data["sample_cadidates"][pred_index]["cui_id"]
                if pred_cui == golden_cui:
                    acc += 1
                num += 1
        w_out.close()
        print("test acc is :", acc / num)


if __name__ == "__main__":
    # hard template predict
    """
    pmt = PromptMedicinePredict(classes=["negative", "positive"],
                                test_data_path = "../data/prompt_for_test",
                                label_words = {"negative": ["no"], "positive": ["yes"]},
                                use_cuda = True,
                                plm_name="bert",
                                plm_path="../pretrain/pubmed_bert",
                                template_text='{"meta": "mention_context"} {"meta": "mention_name"} and the '
                                              '{"meta":"cui_type"} {"meta":"cui_name"} have the same meaning, '
                                              'is it correct? {"mask"}.',
                                load_model_path="../old_model_saved/template_hard/best_9.th",
                                template_name="manual",
                                verbalizer_name="manual")
    """
    # mix template predict
    pmt = PromptMedicinePredict(classes=["negative", "positive"],
                                test_data_path = "../data/top100/medmention.prompt.test.100",
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
    pmt.predict()
