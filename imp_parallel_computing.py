import logging
import math
import time
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

__MAX_WORKERS = 16
__MAX_LINES = 300  # one thread calculate __MAX_LINES lines


def __gene_index(file_path: str) -> Tuple[list, int]:
    logging.info("begin to generate index...")
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()

    sentence_num = len(lines)
    sentence_lens = [0 for _ in range(sentence_num)]
    index = [dict() for _ in range(sentence_num)]
    word_idf = dict()  # record the number of sentences cover the word
    # word frequency statistics
    for i in range(sentence_num):
        words = lines[i].split(' ')
        sentence_lens[i] = len(words)  # record the number of words in a sentence
        if sentence_lens[i] == 0:
            sentence_lens[i] = 1  # avoid division by zero
        for word in words:
            if word not in index[i]:
                index[i][word] = 1
            else:
                index[i][word] += 1
        for word in set(words):
            if word not in word_idf:
                word_idf[word] = 1
            else:
                word_idf[word] += 1

    # convert to tf-idf matrix
    for word in word_idf:  # get idf
        word_idf[word] = math.log(sentence_num / word_idf[word], 10) + 1

    for i in range(sentence_num):
        for word in index[i]:
            tf = index[i][word] / sentence_lens[i]
            index[i][word] = tf * word_idf[word]

    sen_modu = [0 for _ in range(sentence_num)]

    # get the modulus length of sentences
    def get_modu(index, start_id, modu_arr, sen_num):
        if start_id >= sen_num:
            return
        end_id = start_id + __MAX_LINES
        if end_id > sen_num:
            end_id = sen_num
        for sen_id in range(start_id, end_id):
            for word in index[sen_id]:
                modu_arr[sen_id] += index[sen_id][word] ** 2
            modu_arr[sen_id] = math.sqrt(modu_arr[sen_id])
            for word in index[sen_id]:
                index[sen_id][word] /= modu_arr[sen_id]  # normalize

    with ThreadPoolExecutor(max_workers=__MAX_WORKERS) as thread:
        tasks = []
        for i in range(int(sentence_num / __MAX_LINES)):
            tasks.append(thread.submit(get_modu, index, i * __MAX_LINES, sen_modu, sentence_num))
        wait(tasks, return_when=ALL_COMPLETED)

    logging.info("finish index generating")
    return index, sentence_num


def cal_similarity(index: list, sen_num: int) -> dict:
    with open('data/imp_para_result.txt', 'w', encoding='utf-8') as out:
        # go through and compute all the multiplications
        def get_similarity(index, start_id, sen_num, last_task):
            if start_id >= sen_num:
                return
            end_id = start_id + __MAX_LINES
            if end_id > sen_num:
                end_id = sen_num
            simi_matrix = list()
            for i in range(start_id, end_id):
                simi_matrix.append([0 for _ in range(i + 1)])
            for i in range(start_id, end_id):
                for j in range(i + 1):
                    common_keys = set(index[i].keys()).intersection(set(index[j].keys()))
                    for word in common_keys:
                        simi_matrix[i - start_id][j] += index[i][word] * index[j][word]
            if last_task is not None:
                wait([last_task], return_when=ALL_COMPLETED)
            for line in simi_matrix:
                out.write(str(line) + "\n")

        with ThreadPoolExecutor(max_workers=__MAX_WORKERS) as thread:
            last_task = thread.submit(get_similarity, index, 0, sen_num, None)
            for start_id in range(1, int(sen_num / __MAX_LINES)):
                last_task = thread.submit(get_similarity, index, start_id * __MAX_LINES, sen_num, last_task)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    start = time.time()
    index, sen_num = __gene_index("data/data.txt")
    cal_similarity(index, sen_num)
    end = time.time()
    logging.info(f"time cost: {end - start} s")
