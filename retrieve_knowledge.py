import colbert
from colbert import Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
import pickle
import json
import csv


def load_index_mapping(index_file):
    """
    加载 index_queries_usmle.tsv 文件，返回一个字典，
    键为 query_id（第一列），值为 json_index（第二列）。
    """
    index_mapping = {}
    with open(index_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            query_id, json_index = map(int, row)
            index_mapping[query_id] = json_index
    return index_mapping


if __name__ == "__main__":
    checkpoint = 'colbert-ir/colbertv1.9'
    nbits = 2  # encode each dimension with 2 bits
    doc_maxlen = 512  # truncate passages at 300 tokens

    index_name = "medcorp"
    with open('/data/data_user_alpha/MedRAG/corpus/medcorp_chunks.pickle', 'rb') as file:
        collection = pickle.load(file)

    experiment_name = "/data/data_user_alpha/MedRAG/corpus/experiments/notebook"
    query_file = './output/queries_usmle_knowledge.tsv'
    index_file = './output/index_queries_usmle.tsv'
    json_file = './output/usmle_test_ir.json'
    output_file = './output/usmle_test_knowledge1.json'

    # 加载 index 映射关系
    index_mapping = load_index_mapping(index_file)

    # 加载原始 JSON 数据
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 初始化 ColBERT 搜索器
    with Run().context(RunConfig(nranks=1, experiment=experiment_name)):
        config = ColBERTConfig(
            root="./",
            doc_maxlen=512,
            query_maxlen=128,
            nbits=nbits
        )
        searcher = Searcher(index=index_name, checkpoint=checkpoint, config=config)
        import ipdb
        ipdb.set_trace()
        queries = Queries(query_file)
        num_doc = 10
        
        ranking = searcher.search_all(queries, k=num_doc)
        ranks_result = ranking.tolist()

    # 构建最终输出
    output = []

    # 遍历 queries 和 index_mapping，确保正确顺序
    for query_id in queries.data.keys():
        if query_id in index_mapping:  # 查询 ID 存在于映射中
            json_index = index_mapping[query_id]

            # 获取当前 query 的排名结果
            doc_list = []
            start_idx = (query_id - 1) * num_doc
            end_idx = query_id * num_doc
            for x in ranks_result[start_idx:end_idx]:
                doc_list.append(collection[x[1]])

            # 将文档和对应 JSON 数据组合
            input_data = data[json_index - 1]  # JSON 文件索引从 1 开始
            output.append({
                'input': input_data['input'],
                'output': input_data['output'],
                'doc': doc_list
            })

    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=4)

    print(f"Processed and saved output to {output_file}")
