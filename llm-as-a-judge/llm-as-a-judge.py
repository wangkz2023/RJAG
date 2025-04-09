# -*- coding: UTF-8 -*- #

import time

from openai import OpenAI


def load_data(file_path, start=0, end=None):
    """读取数据文件，解析 query 和 document"""
    queries, passages = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        selected_lines = lines[start:end] if end else lines[start:]
        for line in selected_lines:
            parts = line.strip().split(' [SEP] ')
            if len(parts) == 2:
                query, document = parts
                queries.append(query)
                passages.append(document)
    return queries, passages


def judge_relevance(query, document):
    """使用大语言模型判断 query 和 document 的相关性"""
    prompt = f"""
    For the following query and document, judge whether they are “Highly Relevant”, “Somewhat Relevant”, or “Not Relevant”.
    Direct output Relevance, no explanation required.
    Query: {query} 
    Document: {document} 
    Output:
    """
    client = OpenAI(
        # openai系列的sdk，包括langchain，都需要这个/v1的后缀
        base_url='https://api.deepseek.com',
        api_key='your-apikey',
    )
    result = client.chat.completions.create(
        model="deepseek-chat",  # 使用 DeepSeek-V3 模型
        messages=[{"role": "user", "content": prompt}],  # 设置用户的输入消息
        max_tokens=512,  # 设置最大 token 数量
        n=1,
        stop=["null"],
        temperature=0.0  # 让模型尽可能确定性输出
    )
    return result.choices[0].message.content.strip()


def main():
    file_path = "data/popqa/test_pubqa.txt"
    output_file = "data/popqa/popqa_deepseek_as_a_judge_all.txt"
    start, end = 0, 20000  # 控制输入的范围
    queries, passages = load_data(file_path, start, end)
    i = start

    print("The length of queries:", len(queries))
    print("The length of passages:", len(passages))
    print("*********************************")

    with open(output_file, 'a', encoding='utf-8') as f:
        for query, document in zip(queries, passages):
            relevance = judge_relevance(query, document)
            i = i + 1
            print("the " + str(i) + "th" + " document!")
            print(f"Query: {query}\nDocument: {document}\nRelevance: {relevance}\n")
            f.write(f"Query: {query}\nDocument: {document}\nRelevance: {relevance}\n\n")

            # 在每次写入后刷新文件
            f.flush()  # 确保数据立刻写入文件
            # time.sleep(2)

    print("*"*100)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
