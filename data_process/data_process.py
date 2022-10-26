# coding: utf-8

import json
import logging
import random
logging.basicConfig(level=logging.INFO)


class TrainDataGener(object):
    """
    normal train data
    """
    def __init__(self):
        self.data_path = "medmentions_train.txt"
        self.golden_ent_file = "umls2017aa_reference_ont.txt"
        self.golden_type_file = "MRSTY.RRF"
        self.type_file = "SRDEF"
        self.load()

    def load(self):
        self.golden_ent_id_names = dict()
        num_ent = 0
        with open(self.golden_ent_file) as f:
            for line in f:
                ent_id, ent_name = line.strip().split("||")
                ent_id = ent_id.strip()
                ent_name = ent_name.strip()
                num_ent += 1
                if ent_id in self.golden_ent_id_names:
                    self.golden_ent_id_names[ent_id].append(ent_name)
                else:
                    self.golden_ent_id_names[ent_id] = [ent_name]
        logging.info("golden ent_id number is: %d" % (num_ent))

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
        self.type_id_name["UnknownType"] = {"type_name": "unknown type", "type_desc": "unknown type", "type_id": type_id}

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

        result = content[sent_begin+1: sent_end-1]
        return result

    def main(self):
        num_no_golden = 0
        num_unknown_type = 0
        with open(self.data_path) as f:
            line_content = f.readline()
            doc_id_len = len(line_content.split("|")[0])
            title = line_content[doc_id_len+3: ]
            abstract = f.readline()[doc_id_len+3: ]
            for line in f:
                if not line.strip():
                    title = ""
                    abstract = ""
                    continue
                else:
                    if not title and not abstract:
                        doc_id_len = len(line.split("|")[0])
                        title = line[doc_id_len+3: ]
                        abstract = f.readline()[doc_id_len+3: ]
                    else:
                        train_item = {"doc_id": "", "mention_name": "", "mention_context": "",
                                      "cui_id": "", "cui_name": [], "cui_type": []}
                        content = title + abstract
                        doc_id, mention_begin, mention_end, \
                        mention_name, mention_type_ids, ent_id = line.strip().split("\t")
                        if content[int(mention_begin): int(mention_end)] != mention_name:
                            logging.info("content: %s =========%s-%s===" % (content, mention_begin, mention_end))
                            logging.info("doc_id-%s, %s: %s" % (doc_id, content[int(mention_begin): int(mention_end)], mention_name))
                            raise RuntimeError("mention name != content[mention_begin: mention_end] !!")
                        train_item["doc_id"] = doc_id
                        train_item["mention_name"] = mention_name
                        train_item["mention_context"] = self.parse_context(content, mention_begin, mention_end)
                        train_item["cui_id"] = ent_id
                        if ent_id not in self.golden_ent_id_names:
                            num_no_golden += 1
                            continue
                        train_item["cui_name"] = self.golden_ent_id_names[ent_id]
                        mention_type_ids = mention_type_ids.strip().split(",")
                        for cur_type_id in mention_type_ids:
                            cur_type_name = self.type_id_name[cur_type_id]
                            if cur_type_id == "UnknownType":
                                num_unknown_type += 1
                            train_item["cui_type"].append(cur_type_name)
                        print(json.dumps(train_item, ensure_ascii=False))


class PromptData(TrainDataGener):
    def prompt_data(self, prediction_data: str, origin_data: str):
        """
        通过spbert预测结果构造相应训练数据负例，形成最终训练数据
        """
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
                # positive
                for cui_type in origin_json["cui_type"]:
                    for cui_name in origin_json["cui_name"]:
                        if len(res_item["positive"]) > 10:
                            break
                        prompt_example = {"mention_context": mention_context, "mention_name": mention_name,
                                          "cui_type": cui_type["type_name"], "cui_name": cui_name, "label": "positive"}
                        res_item["positive"].append(prompt_example)

                # negative
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

    def prompt_data_global(self, prediction_data: str, origin_data: str):
        """
        prompt train data
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
                candidates = predict_item["mentions"][0]["candidates"]

                cand_list = []
                for item in candidates:
                    cand_list.append(item["labelcui"])

                for candidata in candidates:
                    cui = candidata["labelcui"]
                    neg_cui_name = candidata["name"]
                    for neg_cui_type_id in self.golden_id_type[cui]:
                        neg_cui_type_name = self.type_id_name[neg_cui_type_id]
                        prompt_example = {"mention_context": mention_context, "mention_name": mention_name,
                                          "cui_type": neg_cui_type_name["type_name"], "cui_name": neg_cui_name,
                                          "cui_id": cui}

                        res_item["sample_cadidates"].append(prompt_example)

                print(json.dumps(res_item, ensure_ascii=False))



if __name__ == "__main__":
    # 生成普通样本
    tdg = TrainDataGener()
    tdg.main()

    # 生成prompt格式样本
    pd = PromptData()

    pd.prompt_data(prediction_data="norm_data/predictions_eval_train_pubmedbert_top10.json",
                    origin_data="norm_data/medmention.train")

    # 生成prompt格式待预测样本
    pd.prompt_data_for_test(prediction_data="norm_data/predictions_eval_test.json",
                             origin_data="norm_data/medmention.test")
