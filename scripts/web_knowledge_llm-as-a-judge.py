import argparse
import os
from tqdm import tqdm
import json
from utils import extract_keywords
import requests
from bs4 import BeautifulSoup
import openai


def generate_knowledge_q(questions, task, openai_key, mode):
    if task == 'bio':
        queries = [q[7:-1] for q in questions]
    else:
        queries = extract_keywords(questions, task, openai_key)
    if mode == 'wiki':
        search_queries = ["Wikipedia, " + e for e in queries]
    else:
        search_queries = queries
    return search_queries


def Search(queries, search_path, search_key):
    url = "https://google.serper.dev/search"
    responses = []
    search_results = []
    for query in tqdm(queries, desc="Searching for URLs..."):
        payload = json.dumps({"q": query})
        headers = {'X-API-KEY': search_key, 'Content-Type': 'application/json'}

        reconnect = 0
        while reconnect < 3:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                break
            except requests.exceptions.RequestException:
                reconnect += 1
                print(f'URL: {url} failed * {reconnect}')

        result = json.loads(response.text)
        results = result.get("organic", [])[:10] if "organic" in result else query
        responses.append(results)
        search_results.append({"queries": query, "results": results})

    if search_path != 'None':
        with open(search_path, 'w') as f:
            json.dump(search_results, f, indent=4)

    return search_results


def test_page_loader(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1').text if soup.find('h1') else ""
        paragraphs = [p.text for p in soup.find_all('p') if len(p.text) > 10]
        return [title + ": " + p for p in paragraphs]
    except:
        return []


def query_llm_for_relevance(question, texts, openai_key):
    openai.api_key = openai_key
    prompt = f"Question: {question}\nWhich of the following texts are relevant to the question? Return up to five of the most relevant parts:\n" + "\n".join(
        texts)
    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[{"role": "system", "content": "You are an intelligent assistant"},
                                 {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()


def visit_pages(questions, web_results, output_file, openai_key, mode):
    output_results = []
    progress_bar = tqdm(range(len(questions)), desc="Processing pages...")

    for i, (query, result) in enumerate(zip(questions, web_results)):
        urls = [page["link"] for page in result["results"][:5] if mode != "wiki" or "wikipedia" in page["link"]]
        snippets = [page.get("snippet", page["title"]) for page in result["results"][:5]]

        content = []
        for url in urls:
            content.extend(test_page_loader(url))

        if content:
            results = query_llm_for_relevance(query, content, openai_key)
        else:
            results = '; '.join(snippets)

        output_results.append(results.replace('\n', ' '))
        progress_bar.update(1)

    with open(output_file, 'w') as f:
        f.write('#' + '\n#'.join(output_results))

    return output_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_queries', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--search_key', type=str)
    parser.add_argument('--task', type=str, choices=['popqa', 'pubqa', 'arc_challenge', 'bio'])
    parser.add_argument('--search_path', type=str, default="None")
    parser.add_argument('--mode', type=str, default="wiki", choices=['wiki', 'all'])
    args = parser.parse_args()

    os.environ["OPENAI_API_KEY"] = args.openai_key
    with open(args.input_queries, 'r') as query_f:
        questions = [q.strip() for q in query_f.readlines()][:10]

    search_queries = generate_knowledge_q(questions, args.task, args.openai_key, args.mode)
    search_results = Search(search_queries, args.search_path, args.search_key)
    results = visit_pages(questions, search_results, args.output_file, args.openai_key, args.mode)


if __name__ == '__main__':
    main()
