# coding: utf-8

import torch
import json
import random
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, SoftTemplate, MixedTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from typing import Dict, List, Any
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from sklearn.metrics import precision_recall_fscore_support


class PromptMedicineTrainer(object):
    """
    医疗实体链接prompt训练
    """
    def __init__(self,
                 classes: List[str],
                 train_data_path: str,
                 dev_data_path: str,
                 label_words: Dict[str, List],
                 save_model_dir: str,
                 use_cuda: bool = True,
                 plm_name: str = "bert",
                 plm_path: str = "bert-base-cased",
                 template_text: str = '{"meta": "mention_context"} {"meta": "mention_name"} and the '
                                      '{"meta":"cui_type"} {"meta":"cui_name"} have the same meaning, '
                                      'is it correct? {"mask"}.',
                 epoch_num: int = 20,
                 batch_size: int = 8,
                 load_model_path: str = "",
                 template_name: str="manual",
                 verbalizer_name: str="manual"):

        self.classes = classes
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.label_words = label_words
        self.save_model_dir = save_model_dir
        self.use_cuda = use_cuda
        self.plm_name = plm_name
        self.plm_path = plm_path
        self.template_text = template_text
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.label_id_map = {"positive": 1, "negative": 0}
        self.load_model_path = load_model_path
        self.template_name = template_name
        self.verbalizer_name = verbalizer_name

    def load_dataset(self, path: str):
        dataset = []
        with open(path) as f:
            for index, line in enumerate(f):
                json_item = json.loads(line.strip())
                guid = index
                meta = {}
                meta["mention_context"] = json_item["mention_context"]
                meta["mention_name"] = json_item["mention_name"]
                meta["cui_type"] = json_item["cui_type"]
                meta["cui_name"] = json_item["cui_name"]
                label = self.label_id_map[json_item["label"]]
                dataset.append(InputExample(guid=guid, meta=meta, label=label))
        random.shuffle(dataset)
        return dataset

    def process_batch(self, inputs: List[Any], tokenizer, promptTemplate, WrapperClass):
        prompt_loader = PromptDataLoader(dataset=inputs, tokenizer=tokenizer,
                                         template=promptTemplate, tokenizer_wrapper_class=WrapperClass,
                                         max_seq_length=256, batch_size=self.batch_size)
        return prompt_loader.__iter__().__next__()
   
    def gener_batch(self, inputs_data: List[Any]):
        for start_idx in range(0, len(inputs_data) - self.batch_size + 1, self.batch_size):
            excerpt = slice(start_idx, start_idx + self.batch_size)
            yield inputs_data[excerpt]

    def train(self):
        # load plm
        plm, tokenizer, model_config, WrapperClass = load_plm(self.plm_name, self.plm_path)
        
        # define template
        if self.template_name == "manual":
            promptTemplate = ManualTemplate(text=self.template_text, tokenizer=tokenizer)
        elif self.template_name == "soft":
            promptTemplate = SoftTemplate(model=plm, text=self.template_text, tokenizer=tokenizer)
        elif self.template_name == "mix":
            promptTemplate = MixedTemplate(model=plm, text=self.template_text, tokenizer=tokenizer)
        else:
            raise RuntimeError("template name is not right !")

        # define verbalizer
        if self.verbalizer_name == "manual":
            promptVerbalizer = ManualVerbalizer(classes=self.classes, label_words=self.label_words, tokenizer=tokenizer)
        elif self.verbalizer_name == "soft":
            promptVerbalizer = SoftVerbalizer(plm=plm, classes=self.classes, label_words=self.label_words, tokenizer=tokenizer)
        else:
            raise RuntimeError("verbalizer name is not right !")

        # define prompt model
        prompt_model = PromptForClassification(template=promptTemplate, plm=plm, verbalizer=promptVerbalizer)
        if self.load_model_path:
            state_dict = torch.load(self.load_model_path)
            prompt_model.load_state_dict(state_dict)
            print("load old model succeeded...")

        # load dataset
        train_dataset = self.load_dataset(self.train_data_path)
        print("load train dataset done...")
        dev_dataset = self.load_dataset(self.dev_data_path)
        dev_dataset = dev_dataset[0: int(len(dev_dataset) / 10)]
        print("load dev dataset done...")

        # loss func
        label_weight = torch.tensor([0.4, 0.6])
        if self.use_cuda:
            label_weight = label_weight.cuda()
        loss_func = torch.nn.CrossEntropyLoss(label_weight)

        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
        #optimizer = torch.optim.AdamW([p for n, p in prompt_model.named_parameters()], lr=1e-6)

        # train
        prompt_model.train()
        if self.use_cuda:
            prompt_model = prompt_model.cuda()

        # loop
        print("strat training...")
        best_f = 0.0
        for epoch in range(self.epoch_num):
            tot_loss = 0
            step = 0
            # random select 1/10 train_dataset:
            random.shuffle(train_dataset)
            train_dataset_light = train_dataset[0: int(len(train_dataset) / 10)]
            print("========== epoch {}: train dataset length is {} ===========".format(epoch, len(train_dataset_light)))
            for batch_data in self.gener_batch(train_dataset_light):
                try:
                    inputs = self.process_batch(batch_data, tokenizer, promptTemplate, WrapperClass)
                except Exception as e:
                    print("max seq limit is 256!!!")
                    continue
                if self.use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                labels = inputs['label']
                p,r,f, _ = precision_recall_fscore_support(labels.cpu().tolist(), preds, average='binary', pos_label=1)
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step % 10 == 0:
                    print("Epoch {}, step {}: average loss: {}, batch p: {}, batch r: {}, batch f: {}"
                          .format(epoch, step, tot_loss / (step + 1), p, r, f, flush=True))
                step += 1

            # Evaluate
            allpreds = []
            alllabels = []
            for batch_data in self.gener_batch(dev_dataset):
                try:
                    inputs = self.process_batch(batch_data, tokenizer, promptTemplate, WrapperClass)
                except Exception as e:
                    print("max seq limit is 256!!!")
                    continue
                if self.use_cuda:
                    inputs = inputs.cuda()
                logits = prompt_model(inputs)
                labels = inputs['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            p,r,f, _ = precision_recall_fscore_support(alllabels, allpreds, average='binary', pos_label=1)
            if f > best_f:
                best_f = f
                torch.save(prompt_model.state_dict(), self.save_model_dir + "/best_%d.th" % (epoch))

            print("evaluate at epoch %d, p is: %f, r is %f, f is %f" % (epoch, p, r, f))


if __name__ == "__main__":
    pmt = PromptMedicineTrainer(classes=["negative", "positive"],
                                train_data_path = "../data/prompt.train",
                                dev_data_path = "../data/prompt.dev",
                                label_words = {"negative": ["no"], "positive": ["yes"]},
                                save_model_dir = "../best_ckpt",
                                use_cuda = True,
                                plm_name="bert",
                                plm_path="../pretrain/pubmed_bert",
                                template_text='{"meta": "mention_context"} {"meta": "mention_name", "shortenable": True} {"soft"} and the '
                                              '{"meta":"cui_type"} {"meta":"cui_name", "shortenable": True} {"soft"} have the same meaning, '
                                              'is it correct? {"mask"} {"soft"}.',
                                epoch_num=15,
                                batch_size=32,
                                load_model_path="",
                                template_name="mix",
                                verbalizer_name="manual")
    pmt.train()
