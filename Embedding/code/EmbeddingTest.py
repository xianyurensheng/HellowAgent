import os
import chromadb
from chromadb.utils import embedding_functions

# 1. 初始化本地客户端（数据存储的文件）
client_path = './Embedding/chroma_db'
client = chromadb.PersistentClient(path=client_path)

# 2. 加载Embedding模型
model_path = os.path.expanduser('~/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2')
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = model_path)

# 3. 创建集合
# 配置说明
# - name：集合名
# - embedding_function：指定使用的Embedding模型
# - metadata：索引类型，相似度算法
# 尝试获取已存在的集合，如果不存在则创建
try:
    collection = client.get_collection(
        name='my_collection',
        embedding_function = embedding_func
    )
    print("✅ 获取已存在的集合")
except:
    collection = client.create_collection(
        name='my_collection',
        embedding_function = embedding_func,
        metadata={"hnsw:space": "cosine"}
    )
    print("✅ 创建新集合")

# 4. 插入数据

# 方法1:自动生成
collection.add(
    ids=['1', '2', '3', '4'],
    documents=[
        '人工智能是未来核心技术',
        '向量数据库用于存储Embeddding向量', 
        'RAG技术结合大模型和向量数据库实现知识库问答',
        'Python是Ai开发的常用语言'
    ],
    metadatas=[
        {"category": "AI", "author": "张三", "level": "基础"}, 
        {"category": "数据库", "author": "张三", "level": "进阶"}, 
        {"category": "AI", "author": "李四", "level": "进阶"},
        {"category": "编程", "author": "李四", "level": "基础"}
    ]
)

# 方法2:手动生成，这样做能够更灵活的控制向量的生成和存储
# my_embedding = [
#     [0.1, 0.2, 0.3 ,... , 0.5],
#     [0.1, 0.2, 0.3 ,... , 0.6],
# ]
# collection.add(
#     ids=['5', '6'],
#     embeddings = my_embedding,
#     documents=[
#         '文本1',
#         '文本2'
#         ],
#     metadatas=[
#         {"source": "模块一"},
#         {"source": "模块二"}
#         ]
# )

# 5. 向量检索
# 条件搜索
print("✅ 条件搜索")
result = collection.query(
    query_texts=['怎么用向量数据库做知识库？'],
    n_results=3,
    where = {"$and": [{"category": "AI"}, {"level": "进阶"}]}
)
# where条件说明：可以限制，也可以不限制

for i in range(len(result['ids'])):
    print(f"查询结果{i+1}:")
    print(f"文本: {result['documents'][i]}")
    print(f"相似度: {result['distances'][i]}")
    print(f"元数据: {result['metadatas'][i]}")
    print('-------------------')

# 阈值搜索
print("✅ 阈值搜索")
result = collection.query(
    query_texts=['怎么用向量数据库做知识库？'],
    n_results=3
)
threshold = 0.5
filtered = [
    (doc, dist, meta) 
    for doc, dist, meta in zip(
        result['documents'][0] , 
        result['distances'][0], 
        result['metadatas'][0]
    ) 
    if dist < threshold
]
for item in filtered:
    print(f"文本：{item[0]}，距离：{item[1]:.3f}")

print('-------------------')

# 6. 其他操作
# # 查看所有
# print(client.list_collections())
# # 查看数量
# print(collection.count())
# # 获取所有
# data = collection.get()
# # 删除
# del_id = ['', '']
# collection.delete(id=del_id)
# # 删除所有
# del_collection =""
# collection.delete_collection(name=del_collection)
# # 清空所有
# collection.delete()
