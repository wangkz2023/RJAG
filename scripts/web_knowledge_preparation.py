import argparse
import os
from tqdm import tqdm
import json
from utils import extract_keywords, select_relevants

import requests
from bs4 import BeautifulSoup

from transformers import T5ForSequenceClassification, T5Tokenizer

import sys


# 根据输入的查询生成搜索关键字
def generate_knowledge_q(questions, task, openai_key, mode):
    if task == 'bio':
        queries = [q[7:-1] for q in questions]
    else:
        queries = extract_keywords(questions, task, openai_key)  # 关键词提取
    if mode == 'wiki':
        search_queries = ["Wikipedia, " + e for e in queries]  # 限定范围搜索
    else:
        search_queries = queries
    return search_queries


# 基于给定的查询进行网络搜索，并收集结果。
def Search(queries, search_path, search_key):
    url = "https://google.serper.dev/search"
    responses = []
    search_results = []
    for query in tqdm(queries[:], desc="Searching for urls..."):
        payload = json.dumps(
            {
                "q": query
            }
        )
        headers = {
            'X-API-KEY': search_key,
            'Content-Type': 'application/json'
        }

        reconnect = 0
        while reconnect < 3:  # 失败重试最多3次
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                break
            except (requests.exceptions.RequestException, ValueError):
                reconnect += 1
                print('url: {} failed * {}'.format(url, reconnect))
        # result = response.text
        result = json.loads(response.text)
        if "organic" in result:
            results = result["organic"][:10]  # 只取前10条搜索结果
        else:
            results = query
        responses.append(results)

        search_dict = [{"queries": query, "results": results}]
        search_results.extend(search_dict)
    if search_path != 'None':
        with open(search_path, 'w') as f:
            output = json.dumps(search_results, indent=4)
            f.write(output)
    return search_results

# 网页内容提取
def test_page_loader(url):
    import requests
    from bs4 import BeautifulSoup
    import signal
    def handle(signum, frame):
        raise RuntimeError

    reconnect = 0
    while reconnect < 3:
        try:
            signal.signal(signal.SIGALRM, handle)
            signal.alarm(180)  # 超时限定180秒
            response = requests.get(url)
            break
        except (requests.exceptions.RequestException, ValueError, RuntimeError):
            reconnect += 1
            print('url: {} failed * {}'.format(url, reconnect))
            if reconnect == 3:
                return []
    try:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')  # 解析html
    except:
        return []
    if soup.find('h1') is None or soup.find_all('p') is None:
        return []
    paras = []
    title = soup.find('h1').text   # 提取标题
    paragraphs = soup.find_all('p')   # 提取所有段落
    for i, p in enumerate(paragraphs):
        if len(p.text) > 10:
            paras.append(title + ': ' + p.text)
    return paras


# 访问网页并提取关键信息
def visit_pages(questions, web_results, output_file, model_name, device, mode):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForSequenceClassification.from_pretrained(model_name, output_hidden_states=True)
    top_n = 5
    titles = []
    urls = []
    snippets = []
    queries = []
    for i, result in enumerate(web_results[:]):
        title = []
        url = []
        snippet = []
        if type(result["results"]) == list:
            for page in result["results"][:5]:
                if mode == "wiki":
                    if "wikipedia" in page["link"]:
                        title.append(page["title"])
                        url.append(page["link"])
                else:
                    title.append(page["title"])
                    url.append(page["link"])
                if "snippet" in page:
                    snippet.append(page["snippet"])
                else:
                    snippet.append(page["title"])
        else:
            titles.append([])
            urls.append([])
            snippets.append([result["results"]])
            queries.append(result["queries"])
            continue
        titles.append(title)
        urls.append(url)
        snippets.append(snippet)
        queries.append(result["queries"])
    output_results = []
    progress_bar = tqdm(range(len(questions[:])), desc="Visiting page content...")
    assert len(questions) == len(urls), (len(questions), len(urls))
    i = 0
    for title, url, snippet, query in zip(titles[i:], urls[i:], snippets[i:], queries[i:]):
        if url == []:
            results = '; '.join(snippet)
        else:
            strips = []
            for u in url:
                strips += test_page_loader(u)
            if strips == []:
                output_results.append('; '.join(snippet))
                results = '; '.join(snippet)
            else:
                results, idxs = select_relevants(
                    strips=strips,
                    query=questions[i],
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    top_n=top_n,
                    mode="cross_encode"
                )
        i += 1
        output_results.append(results.replace('\n', ' '))
        progress_bar.update(1)
    with open(output_file, 'w') as f:
        f.write('#')
        f.write('\n#'.join(output_results))
    return output_results


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
        questions = [q.strip() for q in query_f.readlines()][:10]

    search_queries = generate_knowledge_q(questions, args.task, args.openai_key, args.mode)
    search_results = Search(search_queries, args.search_path, args.search_key)
    results = visit_pages(questions, search_results, args.output_file, args.model_path, args.device, args.mode)


if __name__ == '__main__':
    main()
