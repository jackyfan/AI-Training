#导入相关依赖
from langchain_experimental.llms.anthropic_functions import prompt
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

from langchain_ollama.embeddings import OllamaEmbeddings

#设置相关文档目录
KG_DIR= "KG"
#配置匹配文件数量
KG_INDEX=3

#加载文档
print("正在加载文档中。。。")
documents = []

for file_name in os.listdir(KG_DIR):
    if file_name.endswith(".txt"):
        with open(os.path.join(KG_DIR, file_name), "r", encoding="utf-8") as f:
            documents.append(f.read())

print(f'共加载{len(documents)}个文件')

#将文档向量化
print("正在将文档向量化中。。。")
embedding_model = OllamaEmbeddings(model='bge-m3:567m')

vector_store=FAISS.from_texts(documents,embedding_model)

print('初始化推理模型中。。。')
llm = OllamaLLM(model="deepseek-r1:1.5b", streaming=True)


print('知识库初始化完成')
query = input('请输入你的问题：')
docs = vector_store.similarity_search(query=query,k=KG_INDEX)
context_text='\n'.join([doc.page_content for doc in docs])
prompt = f'根据以下的内容回答问题：\n{context_text}\n问题：{query}'

print('模型回答：')
for token in llm.stream(prompt):
    print(token, end='', flush=True)