import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model_path = os.path.expanduser('~/.cache/modelscope/hub/models/sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer(model_path)

sentences = ['哈基米', '不想上班', '猫']

emb1 = model.encode([sentences[0]])
emb2 = model.encode([sentences[1]])
emb3 = model.encode([sentences[2]])

print(f"句子1的向量长度: {emb1.shape[1]}")
print(f"句子2的向量长度: {emb2.shape[1]}")
print(f"句子3的向量长度: {emb3.shape[1]}")

# 计算余弦相似度
sim_1_2 = cosine_similarity(emb1, emb2)[0][0]
sim_1_3 = cosine_similarity(emb1, emb3)[0][0]
sim_2_3 = cosine_similarity(emb2, emb3)[0][0]

print(f"句子1和句子2的相似度: {sim_1_2:.4f}")
print(f"句子1和句子3的相似度: {sim_1_3:.4f}")
print(f"句子2和句子3的相似度: {sim_2_3:.4f}")