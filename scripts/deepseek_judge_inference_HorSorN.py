# -*- coding: UTF-8 -*- #
import argparse
import logging

import os
import re
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_scheduler

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

import time


# 任务说明
TASK_INST = {
    "wow": "Given a chat history separated by new lines, generates an informative, knowledgeable and engaging response. ",
    "pubqa": "Is the following statement correct or not? Say true if it's correct; otherwise say false.",
    "eli5": "Provide a paragraph-length response using simple words to answer the following question.",
    "obqa": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "arc_easy": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "arc_challenge": "Given four answer candidates, A, B, C and D, choose the best answer choice.",
    "trex": "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity.",
    "asqa": "Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers."}
control_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]",
                  "[Retrieval]",
                  "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]",
                  "[Utility:3]", "[Utility:4]", "[Utility:5]"]
'''
task = ### Instruction: + TASK_INST[task] + ## Input: + question + ### Response:
task = ### Instruction: + TASK_INST[task] + ## Input: + question + choices + ### Response:
'''


# 根据给定的任务类型（task）、问题（question）和段落（paragraph），生成一个适合该任务的提示文本（prompt）
def format_prompt(task, question, paragraph=None):
    if paragraph is not None:
        paragraph = ' '.join(paragraph.split(' ')[:])

    instruction = TASK_INST[task] if task in TASK_INST else None
    instruction = instruction + "\n\n## Input:\n\n" + question if instruction is not None else question

    if instruction == question:  # 如果 instruction 仍然等于 question（即没有提供任务说明）
        # PopQA
        prompt = "Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: " + paragraph + "\n\nInstruction: Answer the question: " + question
        # Refer to the following documents, follow the instruction and answer the question.
        #
        # Documents: [the paragraph]
        #
        # Instruction: Answer the question: [the question]

    return prompt

# 根据给定的任务类型（task）、问题（question）和段落（paragraph），生成一个适合该任务的提示文本（prompt）
def format_prompt(i, task, question, paragraph=None, modelname="selfrag_llama"):
    if paragraph is not None:
        paragraph = ' '.join(paragraph.split(' ')[:])

    instruction = TASK_INST[task] if task in TASK_INST else None
    instruction = instruction + "\n\n## Input:\n\n" + question if instruction is not None else question

    if task == "arc_challenge":
        with open("../data/arc_challenge/choices", 'r') as f:
            choices = f.readlines()[i].strip()
        choices = choices.replace("A: ", "\nA: ")
        choices = choices.replace("B: ", "\nB: ")
        choices = choices.replace("C: ", "\nC: ")
        choices = choices.replace("D: ", "\nD: ")
        choices = choices.replace("E: ", "\nE: ")
        instruction += choices

    if instruction == question:  # 如果 instruction 仍然等于 question（即没有提供任务说明）
        # PopQA
        # prompt = "Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: " + paragraph + "\n\nInstruction: Answer the question: " + question
        prompt = "Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: " + paragraph + "\n\nInstruction: Answer the question: " + question
        # Refer to the following documents, follow the instruction and answer the question.
        #
        # Documents: [the paragraph]
        #
        # Instruction: Answer the question: [the question]
    else:
        if task == "arc_challenge":
            prompt = "Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: " + paragraph + "\nQuestion: " + question + "\n\nInstruction: Given four answer candidates, A, B, C and D, choose the best answer choice." + "\nChoices:" + choices

        elif task == "pubqa":
            if modelname == "llama":
                prompt = "Read the documents and answer the question: Is the following statement correct or not? \n\nDocuments: " + paragraph + "\n\nStatement: " + question + "\n\nOnly say true if the statement is true; otherwise say false."
            else:
                prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
                if paragraph is not None:
                    prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
            # prompt = "Refer to the following documents, follow the instruction and answer the question.\n\nDocuments: " + paragraph + "\n\nInstruction: Is the following statement correct or not? Say true if it's correct; otherwise say false. \nStatement: " + question
    # prompt = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
    # if paragraph is not None:
    #     prompt += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)

    return prompt


# 这个函数用于后处理模型的输出，去除一些控制符和多余的标记
def postprocess_answer_option_conditioned(answer):
    for token in control_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer


# 这个函数负责读取并预处理输入数据（文件），将查询和相关的文档片段提取出来
# 返回处理后的问题列表（queries）和段落列表（passages）
"""
有"Highly Relevant"只使用"Highly Relevant"的文本;
没有"Highly Relevant"的使用"Somewhat Relevant"的文本。
其他的使用"Not Relevant"代替
"""

def data_preprocess(file_path, n_docs):
    # 定义有效的 Relevance 值
    valid_relevance = "Highly Relevant"
    valid_relevance2 = "Somewhat Relevant"
    valid_relevance3 = "Not Relevant"
    # valid_relevance = ["Highly Relevant"]

    # 打开文件并读取内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按照空行分割每一组数据
    entries = content.split("\n\n")

    # 初始化存储 Query 和 Document 的列表
    queries = []
    documents = []
    query_only = None
    passage_list = []
    passage_list2 = []
    passage_list3 = []

    for entry in entries:

        # 使用正则表达式提取 Query, Document 和 Relevance 字段
        query_match = re.search(r"Query: (.*?)\n", entry)
        document_match = re.search(r"Document: (.*?)\n", entry)
        relevance_match = re.search(r"Relevance: (.*?)\s*$", entry)

        # 如果找到了 Query、Document 和 Relevance
        if query_match and document_match and relevance_match:
            query = query_match.group(1).strip()
            document = document_match.group(1).strip()
            relevance = relevance_match.group(1).strip()

        if query_only is None:
            # print(11111)
            query_only = query
            # 如果 Relevance 是 "Highly Relevant"
            if relevance == valid_relevance:
                passage_list.append(document)
            # 如果 Relevance 是 "Somewhat Relevant"
            if relevance == valid_relevance2:
                passage_list2.append(document)
                # 如果 Relevance 是 "No Relevant"
            if relevance == valid_relevance3:
                passage_list3.append(document)
        elif query_only == query:
            if relevance == valid_relevance:
                passage_list.append(document)
            if relevance == valid_relevance2:
                passage_list2.append(document)
            if relevance == valid_relevance3:
                passage_list3.append(document)
        else:
            queries.append(query_only)
            # print(query_only)
            if passage_list:
                documents.append("\n".join(passage_list))
            elif passage_list2:
                # print(query_only)
                documents.append("\n".join(passage_list2))
            elif passage_list3:
                documents.append("\n ".join(passage_list3))
                # print(query_only)
            else:
                documents.append("None")

            passage_list = []
            passage_list2 = []
            passage_list3 = []
            query_only = query

            if relevance == valid_relevance:
                passage_list.append(document)
            if relevance == valid_relevance2:
                passage_list2.append(document)
            if relevance == valid_relevance3:
                passage_list3.append(document)

    # 处理最后一个
    queries.append(query_only)
    if passage_list:
        documents.append("\n".join(passage_list))
    elif passage_list2:
        documents.append("\n".join(passage_list2))
    elif passage_list3:
        documents.append("\n".join(passage_list3))
    else:
        documents.append("None")

    none_count = documents.count("None")
    print(none_count)
    return queries, documents


# 该函数从文件中读取评估数据，并根据是否有标签的情况进行处理
def get_evaluator_data(file):
    with_label = False
    # with_label = True
    content = []
    label = []
    with open(file, "r", encoding="utf-8") as f:
        if with_label:
            for line in f.readlines()[:]:
                c, l = line.split("\t")
                content.append(c)
                label.append((int(l.strip()) - 0.5) * 2)
            return content, label
        else:
            for line in f.readlines():
                content.append(line.strip())
            return content, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--download_dir', type=str, help="specify vllm model download dir",
                        default=".cache")
    parser.add_argument("--ndocs", type=int, default=-1,
                        help="Number of documents to retrieve per questions")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of documents to retrieve per questions")

    args = parser.parse_args()

    generator = LLM(model=args.generator_path, dtype="half")
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100, skip_special_tokens=False)


    queries, passages = data_preprocess(args.input_file, args.ndocs)

    paragraphs = passages

    preds = []
    modelname = "llama"
    for i, (q, p) in tqdm(enumerate(zip(queries, paragraphs))):
        prompt = format_prompt(i, args.task, q, p, modelname)
        pred = generator.generate([prompt], sampling_params)
        preds.append(postprocess_answer_option_conditioned(pred[0].outputs[0].text))

    with open(args.output_file, 'w') as f:
        f.write('\n'.join(preds))


if __name__ == '__main__':
    main()
