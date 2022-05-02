# coding: utf-8

import json
import logging
import random
logging.basicConfig(level=logging.INFO)
import faiss
import numpy as np
import sys


class VectorSearch(object):
    """
    对prompt模型生成的mention向量检索top n相似entity向量生成候选
    """
    def __init__(self,
                 entity_vector_path:
                 str="../data/soft_template_entity_vector.pubmed",
                 mention_data_path:
                 str="../data/soft_template_mention_vector.pubmed",
                 buffer_size: int = 20000,
                 out_path = "../data/top100_result_final.pubmed"):
        self.entity_vector = None
        self.entity_ids = None
        self.id_to_cui = dict()
        self.mention_data = None
        self.index = faiss.IndexFlatL2(768)
        self.index_2 = faiss.IndexIDMap(self.index)
        #res = faiss.StandardGpuResources() 
        #res.setTempMemory(512 * 1024 * 1024 * 1024 * 1024)
        #self.index_2 = faiss.index_cpu_to_all_gpus(self.index_2)
        #logging.info("load gpu done")
        self.entity_vector_path = entity_vector_path
        self.mention_data_path = mention_data_path
        self.buffer_size = buffer_size
        self.out_path = out_path
        self.load()

    def load(self):
        batch_entity_vector = []
        batch_entity_ids = []
        indexed_num = 0
        with open(self.entity_vector_path) as f:
            for num, line in enumerate(f):
                try:
                    json_data = json.loads(line.strip())
                except Exception as e:
                    print("error!")
                    json_data = {"cui_id": "-1", "cui_name": "-1", "vector":
                            list(np.zeros(768))}
                if len(json_data["vector"]) != 768:
                    json_data = {"cui_id": "-1", "cui_name": "-1", "vector":
                            list(np.zeros(768))}
                cui_id = json_data["cui_id"]
                cui_name = json_data["cui_name"]
                vector = json_data["vector"]
                self.id_to_cui[num] = cui_id
                if num % 2000 == 0:
                    logging.info("index num is : %d" % (num))

                if len(batch_entity_vector) < self.buffer_size:
                    batch_entity_vector.append(vector)
                    batch_entity_ids.append(num)
                else:
                    batch_entity_vector = np.array(batch_entity_vector).astype(np.float32)
                    batch_entity_ids = np.array(batch_entity_ids)
                    indexed_num += len(batch_entity_vector)
                    try:
                        self.index_2.add_with_ids(batch_entity_vector, batch_entity_ids)
                    except Exception as e:
                        print(e)
                        continue
                    batch_entity_vector = []
                    batch_entity_ids = []
                    batch_entity_vector.append(vector)
                    batch_entity_ids.append(num)

        if len(batch_entity_vector) > 0:
            batch_entity_vector = np.array(batch_entity_vector).astype(np.float32)
            batch_entity_ids = np.array(batch_entity_ids)
            self.index_2.add_with_ids(batch_entity_vector, batch_entity_ids)
            indexed_num += len(batch_entity_vector)

        logging.info("load entity vector into index done, num is: %d" % (indexed_num))

    def search(self):
        num_acc = 0
        num = 0
        wo = open(self.out_path, "w")
        batch_data = []
        batch_json = []
        with open(self.mention_data_path) as f:
            for line in f:
                json_data = json.loads(line.strip())
                vector = json_data["vector"]
                if len(batch_data) < 100:
                    batch_data.append(vector)
                    batch_json.append(json_data)
                else:
                    mention_vectors = np.array(batch_data).astype(np.float32)
                    D, I = self.index_2.search(mention_vectors, 100)
                    for json_item, top_k_dis, top_k_res in zip(batch_json, D, I):
                        top_k_cui = list()
                        for recall_dis, recall_index in zip(top_k_dis, top_k_res):
                            top_k_cui.append({"cui_id":
                                self.id_to_cui[recall_index], "id":
                                int(recall_index), "distance": float(recall_dis)})
                        json_item["top_k_result"] = top_k_cui
                        wo.write(json.dumps(json_item, ensure_ascii=False) + "\n")
                    batch_data = []
                    batch_json = []
                    batch_data.append(vector)
                    batch_json.append(json_data)
        if len(batch_data) > 0:
            mention_vectors = np.array(batch_data).astype(np.float32)
            D, I = self.index_2.search(mention_vectors, 100)
            for json_item, top_k_dis, top_k_res in zip(batch_json, D, I):
                top_k_cui = list()
                for recall_dis, recall_index in zip(top_k_dis, top_k_res):
                    top_k_cui.append({"cui_id": self.id_to_cui[recall_index],
                        "id": int(recall_index), "distance": float(recall_dis)})
                json_item["top_k_result"] = top_k_cui
                wo.write(json.dumps(json_item, ensure_ascii=False) + "\n")
        wo.close()



if __name__ == "__main__":
    entity_vector_file = sys.argv[1]
    mention_vector_file = sys.argv[2]
    out_path = sys.argv[3]
    vs = VectorSearch(entity_vector_path=entity_vector_file,
            mention_data_path=mention_vector_file,
            buffer_size=20000,
            out_path=out_path)
    vs.search()

