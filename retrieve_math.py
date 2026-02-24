from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert import Searcher
import torch
import ipdb
import os
import json
import re
import csv
import pandas as pd
import numpy as np
import sys
import tqdm
from datetime import timedelta



os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_TIMEOUT"] = "12000"
def extract_evidence_and_rule(text):
    """
    使用正则表达式提取 Evidence 和 Rule 对，并将其合并为单条记录。
    """
    pattern = r"\d+\.\s*Evidence:\s*(.+?)\s*Rule:\s*(.+?)(?=\d+\.|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    combined_records = []
    for evidence, rule in matches:
        combined_text = f"Evidence: {evidence.strip()}  Rule: {rule.strip()}"
        combined_records.append(combined_text)
    return combined_records

def process_json_files(json_dir, output_csv, all_data=[], index = 0):
    """
    遍历目录下的所有 JSON 文件，提取 'rule' 键中的内容，并保存到 CSV 文件。
    """
    
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, "r") as file:
                try:
                    datas = json.load(file)
                    for data in datas:
                        if "rule" in data and data['rule']:
                            rule_text = data["rule"]
                            all_data.append({"id": index, "combined": rule_text.replace('\n', '  ')})
                            index += 1
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_name}")
    # 将所有数据保存到 CSV 文件
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False, header=False, sep='\t')
    print(f"Extracted data saved to {output_csv}")
    return df, index
if __name__ == "__main__":
    json_dir = "colbert-ir/colbertv1.9"  # JSON 文件目录
    collection = "output/numina_math_inputs.tsv"  # 输出的 CSV 文件路径
    collection2 = "output/numina_math_inputs_answer.tsv"  # 输出的 CSV 文件路径
    qury_json = "./output/amc23_test.json"
    #all_data = []
    #index = 0
    #data, index = process_json_files(json_dir, collection, all_data, index)
    #print(index)
    #data, _ = process_json_files(json_dir, collection, all_data, index)
    
    nbits = 2
    #checkpoint_path = "/home/jwang/Project/uptodata/medical_rules/train_colbert/ColBERT/experiments/default/none/2025-01/14/20.43.13/checkpoints/colbert"
    checkpoint_path = "colbert-ir/colbertv1.9"
    experiment_root = "./experiments"
    experiment_name = "open_rule_math_qa"
    index_name = f'{experiment_name}.{nbits}bits'
    query_file = './output/amc23_test.tsv'
    config = ColBERTConfig(doc_maxlen=512, query_maxlen=128, nbits=nbits)
    """
    with Run().context(RunConfig(nranks=2, root=experiment_root, experiment=experiment_name)):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=512, query_maxlen=128, nbits=nbits)
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)
    
    import ipdb
    ipdb.set_trace()
    """
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        config = ColBERTConfig(
            root="./",
            doc_maxlen=512, 
            query_maxlen=128, 
            nbits=nbits
        )
        searcher = Searcher(index=index_name, checkpoint=checkpoint_path, config=config)
        queries = Queries(query_file)
        #import ipdb
        #ipdb.set_trace()
        
        #rank1 = searcher.search(queries.data[1], k=10)
        #rank1 = searcher.search(text, k=10)


        num_doc = 100
        ranking = searcher.search_all(queries, k=100)
        ranks_result = ranking.tolist()
        output = []

        with open('./output/numina_math_cot.json', 'r') as file:
            data2 = json.load(file)
        with open(qury_json, 'r') as file:
            data = json.load(file)
        for i in range(len(data)):
            doc_list = []
            for x in ranks_result[i*num_doc:(i+1)*num_doc]:
                if data2[x[1]]['input'] in data[i]['input'] or data[i]['input'] in data2[x[1]]['input']:
                    continue
                doc_list.append({'id': x[1], 'input': data2[x[1]]['input'], 'output': data2[x[1]]['output']})
            print(len(doc_list))
            output.append({'input': data[i]['input'], 'output': data[i]['output'], 'doc': doc_list})
        import ipdb
        ipdb.set_trace()
        with open('./output/amc23_test_qa.json', 'w') as file:
            json.dump(output, file, indent=4)

