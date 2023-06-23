# coding: utf-8

import json
import logging
import random
import glob
import os
logging.basicConfig(level=logging.INFO)


class TrainDataGener(object):
    def __init__(self):
        self.data_path = "proc_v2/processed_train"
        self.order_path = "norm_data/ncbi_train_order.txt"
        self.golden_ent_file = "proc_v2/train_dictionary.txt"
        self.txt_path = "proc_v2/train"
        self.load()

    def load(self):
        self.golden_ent_id_names = dict()
        num_ent = 0
        with open(self.golden_ent_file) as f:
            for line in f:
                ent_ids, ent_name = line.strip().split("||")
                ent_ids = ent_ids.strip().split("|")
                ent_name = ent_name.strip()
                num_ent += 1
                for ent_id in ent_ids:
                    if ent_id in self.golden_ent_id_names:
                        self.golden_ent_id_names[ent_id].append(ent_name)
                    else:
                        self.golden_ent_id_names[ent_id] = [ent_name]
        logging.info("golden ent_id number is: %d" % (num_ent))
        self.ent_type = {}
        self.concept_list = []
        with open(self.order_path) as f:
            for line in f:
                concept_name = line.strip().split("/")[-1]
                concept_path = os.path.join(self.data_path, concept_name)
                self.concept_list.append(concept_path)

        for concept_file_path in self.concept_list:
            concept_name = concept_file_path.split("/")[-1][0: -8]
            txt_file_path = os.path.join(self.txt_path, concept_name + ".txt")
            txt_content = open(txt_file_path).readlines()
            txt_content = txt_content[0] + txt_content[2]
            with open(concept_file_path) as f:
                for line in f:
                    line_content = line.strip()
                    doc_id, mention_index, type, mention_name, cui_ids_origin = line_content.split("||")
                    mention_name = mention_name.strip().split("|")[0]
                    cui_ids_origin = cui_ids_origin.strip()
                    if "+" in cui_ids_origin:
                        cui_ids_origin = cui_ids_origin.replace("+", "|")
                    cui_ids = cui_ids_origin.strip().split("|")

                    for cui_id in cui_ids:
                        if cui_id not in self.ent_type:
                            self.ent_type[cui_id] = [type]
                        else:
                            self.ent_type[cui_id].append(type)

    def parse_context(self, content: str, begin: int, end: int):
        sent_begin = int(begin)
        sent_end = int(end)
        tag = [".", "?", "!"]
        while sent_begin >= 0:
            if content[sent_begin] in tag:
                break
            else:
                sent_begin -= 1

        while sent_end <= len(content):
            if sent_end < len(content) and content[sent_end] in tag:
                break
            else:
                sent_end += 1

        result = content[sent_begin+1: sent_end+1]
        return result

    def main(self):
        num_golden_error = 0
        num = 0
        for concept_file_path in self.concept_list:
            concept_name = concept_file_path.split("/")[-1][0: -8]
            txt_file_path = os.path.join(self.txt_path, concept_name + ".txt")
            txt_content = open(txt_file_path).readlines()
            txt_content = txt_content[0] + txt_content[2]
            with open(concept_file_path) as f:
                for line in f:
                    num += 1
                    line_content = line.strip()
                    doc_id, mention_index, type, mention_name, cui_ids_origin = line_content.split("||")
                    mention_name = mention_name.strip().split("|")[0]
                    cui_ids_origin = cui_ids_origin.strip()
                    if "+" in cui_ids_origin:
                        cui_ids_origin = cui_ids_origin.replace("+", "|")
                    cui_ids = cui_ids_origin.strip().split("|")

                    mention_begin, mention_end = mention_index.split("|")
                    mention_begin = int(mention_begin)
                    mention_end = int(mention_end)

                    new_txt_content = txt_content[0: mention_begin] + mention_name + txt_content[mention_end:]
                    mention_end = mention_begin + len(mention_name)
                    if new_txt_content[mention_begin: mention_end] != mention_name:
                        raise RuntimeError("txt_content[mention_begin: mention_end] != mention_name !!!")
                    mention_context = self.parse_context(new_txt_content, mention_begin, mention_end)

                    train_item = {"doc_id": doc_id, "mention_name": mention_name, "mention_context": mention_context,
                                  "cui_id": cui_ids_origin, "cui_name": [], "cui_type": [type]}

                    for cui_id in cui_ids:
                        for name in self.golden_ent_id_names[cui_id]:
                            if name not in train_item["cui_name"]:
                                train_item["cui_name"].append(name)

                    print(json.dumps(train_item, ensure_ascii=False))


class PromptData(TrainDataGener):
    def prompt_data(self, prediction_data: str, origin_data: str):
        """
        通过spbert预测结果构造相应训练数据负例，形成最终训练数据
        """
        num_pos = 0
        num_neg = 0
        num_type = 0
        with open(prediction_data) as f:
            predict_list = json.load(f)["queries"]

        with open(origin_data) as f:
            all_sample_cuis = list(self.golden_ent_id_names.keys())
            random.shuffle(all_sample_cuis)
            start_sample_index = 0
            for origin_item, predict_item in zip(f, predict_list):
                origin_json = json.loads(origin_item.strip())
                assert origin_json["cui_id"] == predict_item["mentions"][0]["golden_cui"]
                assert origin_json["mention_name"] == predict_item["mentions"][0]["mention"]

                res_item = {"positive": [], "negative": []}
                cui_id = origin_json["cui_id"]
                mention_context = origin_json["mention_context"]
                mention_name = origin_json["mention_name"]
                # positive limit 10
                for cui_type in origin_json["cui_type"]:
                    for cui_name in origin_json["cui_name"]:
                        if len(res_item["positive"]) > 30:
                            break
                        prompt_example = {"mention_context": mention_context, "mention_name": mention_name,
                                          "cui_type": cui_type, "cui_name": cui_name, "label": "positive"}
                        res_item["positive"].append(prompt_example)

                # negative
                candidates = predict_item["mentions"][0]["candidates"]
                for candidata in candidates:
                    cui = candidata["labelcui"]
                    cuis = cui.split("|")
                    flag = False
                    for item in cuis:
                        if item in cui_id:
                            flag = True
                            break

                    if not flag:
                        neg_cui_name = candidata["name"]
                        if cuis[0] not in self.ent_type:
                            type_cands = [""]
                        else:
                            type_cands = self.ent_type[cuis[0]]
                        for neg_cui_type_name in type_cands:
                            prompt_example = {"mention_context": mention_context, "mention_name": mention_name,
                                              "cui_type": neg_cui_type_name, "cui_name": neg_cui_name,
                                              "label": "negative"}
                            res_item["negative"].append(prompt_example)

                if len(res_item["negative"]) < 20:
                    need_count = 20 - len(res_item["negative"])
                    if start_sample_index + need_count + 1 < len(all_sample_cuis):
                        select_random_cuis = all_sample_cuis[start_sample_index: start_sample_index + need_count + 1]
                        start_sample_index += (need_count + 1)
                    else:
                        start_sample_index = 0
                        select_random_cuis = all_sample_cuis[start_sample_index: start_sample_index + need_count + 1]
                        start_sample_index += (need_count + 1)

                    for random_cui in select_random_cuis:
                        if random_cui not in cui_id:
                            random_cui_name = self.golden_ent_id_names[random_cui][0]
                            if random_cui not in self.ent_type:
                                random_cui_type = ""
                            else:
                                random_cui_type = self.ent_type[random_cui][0]
                            prompt_example = {"mention_context": mention_context, "mention_name": mention_name,
                                              "cui_type": random_cui_type, "cui_name": random_cui_name,
                                              "label": "negative"}
                            res_item["negative"].append(prompt_example)
                num_pos += len(res_item["positive"])
                num_neg += len(res_item["negative"])
                for k, v in res_item.items():
                    for v_item in v:
                        if v_item["cui_type"] == "":
                            num_type += 1
                        print(json.dumps(v_item, ensure_ascii=False))

    def prompt_data_global(self, prediction_data: str, origin_data: str):
        """
        通过spbert预测结果及全局结果构造相应训练数据负例，形成最终训练数据
        """
        with open(prediction_data) as f:
            predict_list = json.load(f)["queries"]

        with open(origin_data) as f:
            all_sample_cuis = list(self.golden_ent_id_names.keys())
            random.shuffle(all_sample_cuis)
            need_count = 5
            start_sample_index = 0
            for origin_item, predict_item in zip(f, predict_list):
                origin_json = json.loads(origin_item.strip())
                if not origin_json.keys():
                    continue
                assert origin_json["cui_id"] == predict_item["mentions"][0]["golden_cui"]
                assert origin_json["mention_name"] == predict_item["mentions"][0]["mention"]

                res_item = {"positive": [], "negative": []}
                cui_id = origin_json["cui_id"]
                mention_context = origin_json["mention_context"]
                mention_name = origin_json["mention_name"]
                # positive
                for cui_type in origin_json["cui_type"]:
                    for cui_name in origin_json["cui_name"]:
                        prompt_example = {"mention_context": mention_context, "mention_name": mention_name,
                                          "cui_type": cui_type["type_name"], "cui_name": cui_name, "label": "positive"}
                        res_item["positive"].append(prompt_example)

                # locate negative
                candidates = predict_item["mentions"][0]["candidates"]
                for candidata in candidates:
                    cui = candidata["labelcui"]
                    if cui != cui_id:
                        neg_cui_name = candidata["name"]
                        for neg_cui_type_id in self.golden_id_type[cui]:
                            neg_cui_type_name = self.type_id_name[neg_cui_type_id]
                            prompt_example = {"mention_context": mention_context, "mention_name": mention_name,
                                              "cui_type": neg_cui_type_name["type_name"], "cui_name": neg_cui_name,
                                              "label": "negative"}
                            res_item["negative"].append(prompt_example)
                    if len(res_item["negative"]) > 10:
                        break

                # global negative
                if start_sample_index + need_count + 1 < len(all_sample_cuis):
                    select_random_cuis = all_sample_cuis[start_sample_index: start_sample_index+need_count+1]
                    start_sample_index += (need_count+1)
                else:
                    start_sample_index = 0
                    select_random_cuis = all_sample_cuis[start_sample_index: start_sample_index + need_count + 1]
                    start_sample_index += (need_count + 1)

                for random_cui in select_random_cuis:
                    if random_cui != cui_id:
                        random_cui_name = self.golden_ent_id_names[random_cui][0]
                        random_cui_type = self.type_id_name[self.golden_id_type[random_cui][0]]
                        prompt_example = {"mention_context": mention_context, "mention_name": mention_name,
                                          "cui_type": random_cui_type["type_name"], "cui_name": random_cui_name,
                                          "label": "negative"}
                        res_item["negative"].append(prompt_example)
                for k, v in res_item.items():
                    for v_item in v:
                        print(json.dumps(v_item, ensure_ascii=False))

    def prompt_data_for_test(self, prediction_data: str, origin_data: str):
        with open(prediction_data) as f:
            predict_list = json.load(f)["queries"]

        with open(origin_data) as f:
            for origin_item, predict_item in zip(f, predict_list):
                origin_json = json.loads(origin_item.strip())
                if not origin_json.keys():
                    continue
                assert origin_json["cui_id"] == predict_item["mentions"][0]["golden_cui"]
                assert origin_json["mention_name"] == predict_item["mentions"][0]["mention"]

                res_item = {"golden_cui": "", "sample_cadidates": []}
                cui_id = origin_json["cui_id"]
                res_item["golden_cui"] = cui_id
                mention_context = origin_json["mention_context"]
                mention_name = origin_json["mention_name"]

                # candidates sample
                candidates = predict_item["mentions"][0]["candidates"][0: 10]

                for candidata in candidates:
                    cui_id_golden = candidata["labelcui"]
                    cuis = cui_id_golden.strip().split("|")
                    type_cands = []
                    for cui in cuis:
                        if cui in self.ent_type:
                            for item in self.ent_type[cui]:
                                if item not in type_cands:
                                    type_cands.append(item)
                    if not type_cands:
                        type_cands.append("")

                    neg_cui_name = candidata["name"]

                    for neg_cui_type_name in type_cands:
                        prompt_example = {"mention_context": mention_context, "mention_name": mention_name,
                                          "cui_type": neg_cui_type_name, "cui_name": neg_cui_name,
                                          "cui_id": cui_id_golden}

                        res_item["sample_cadidates"].append(prompt_example)

                print(json.dumps(res_item, ensure_ascii=False))


if __name__ == "__main__":
    # 生成普通样本
    tdg = TrainDataGener()
    tdg.main()

    # 生成prompt格式样本
    # pd = PromptData()

    # pd.prompt_data(prediction_data="predictions_eval_ncbi_train.json",
    #                origin_data="ncbi.train")

    # 生成prompt格式待预测样本
    # pd.prompt_data_for_test(prediction_data="predictions_eval_ncbi_test.json",
    #                         origin_data="ncbi.test")




