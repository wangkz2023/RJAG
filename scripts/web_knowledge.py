# -*- coding: UTF-8 -*- #

import argparse
from utils import extract_keywords
import os
from tqdm import tqdm
import json
import requests


# 问题重写
def generate_knowledge_q(questions, task, openai_key, mode, output_file):
    if task == 'bio':
        queries = [q[7:-1] for q in questions]
    else:
        queries = extract_keywords(questions, task, openai_key)
    if mode == 'wiki':
        search_queries = ["Wikipedia, " + e for e in queries]
    else:
        search_queries = queries
    # Save rewritten queries to the specified output file
    with open(output_file, 'a', encoding="utf-8") as f:
        for query in search_queries:
            f.write(query + '\n')
    return search_queries


def Search(queries, search_path, search_key):
    """
    :param queries: 搜索查询列表，每个元素是一个字符串，表示要搜索的关键词。
    :param search_path: 用于存储搜索结果的文件路径，如果为 "None"，则不保存。
    :param search_key: google.serper.dev API 需要的密钥。
    :return:
    """
    url = "https://google.serper.dev/search"
    responses = []  # 存储每个查询的搜索结果列表
    search_results = []  # 存储最终的搜索结果，每个元素包含查询词和对应的搜索结果。
    for query in tqdm(queries[:], desc="Searching for urls..."):  # 用 tqdm 创建进度条，显示搜索进度
        payload = json.dumps(  # 构造 POST 请求的 JSON 负载，包含查询词 q
            {
                "q": query
            }
        )
        headers = {  # 设定请求头
            'X-API-KEY': search_key,
            'Content-Type': 'application/json'  # 指定 JSON 格式的请求数据
        }

        reconnect = 0
        while reconnect < 3:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)  # 发送 POST 请求，查询 query 相关的搜索结果。
                break
            except (requests.exceptions.RequestException, ValueError):  # 处理网络请求异常
                reconnect += 1
                print('url: {} failed * {}'.format(url, reconnect))
        # result = response.text
        result = json.loads(response.text)  # 解析 JSON 响应数据
        if "organic" in result:  # 如果 result 中存在 "organic"（自然搜索结果）
            results = result["organic"][:10]
        else:
            results = query  # 否则，将 results 设为 query（表示未找到结果）
        # 存储搜索结果到 responses
        responses.append(results)
        search_dict = [{"queries": query, "results": results}]
        search_results.extend(search_dict)
    if search_path != 'None':
        with open(search_path, 'w') as f:
            output = json.dumps(search_results, indent=4)
            f.write(output)
    return search_results


def visit_pages(search_results, output_file):
    with open(output_file, "w", encoding="utf-8") as out_f:
        for entry in search_results:
            pos1, pos2 = None, None  # 存储 position 1 和 position 2 的内容

            for result in entry.get("results", []):
                position = result.get("position")
                title = result.get("title", "").strip()
                snippet = result.get("snippet", "").strip()
                content = f"{title}: {snippet}"

                if position == 1:
                    pos1 = content
                elif position == 2:
                    pos2 = content

                # 如果已经找到位置 1 和位置 2，提前结束循环
                if pos1 and pos2:
                    break

            # 组合输出，如果某个位置不存在，则用 "None" 代替
            output_line = f"{pos1 if pos1 else 'None'} [SEP] {pos2 if pos2 else 'None'}\n"
            out_f.write(output_line)

    print(f"提取完成，结果已保存到 {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--input_queries', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--search_key', type=str)
    parser.add_argument('--task', type=str, choices=['popqa', 'pubqa', 'arc_challenge', 'bio'])
    parser.add_argument('--search_path', type=str, default="None")
    parser.add_argument('--mode', type=str, default="wiki", choices=['wiki', 'all'],
                        help="Optional strategies to modify search queries, wiki means web search engine tends to search from Wikipedia")
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.openai_key
    with open(args.input_queries, 'r') as query_f:
        questions = [q.strip() for q in query_f.readlines()]

    search_queries = generate_knowledge_q(questions, args.task, args.openai_key, args.mode)
    search_results = Search(search_queries, args.search_path, args.search_key)
    results = visit_pages(search_results, args.output_file)


if __name__ == '__main__':
    main()
