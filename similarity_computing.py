import logging
import math
import time
from typing import Tuple


def __gene_index(file_path: str) -> Tuple[dict, int]:
    logging.info("begin to generate index...")
    index = dict()
    with open(file_path, encoding='utf-8') as file:
        lines = file.readlines()

    sentence_num = len(lines)
    sentence_lens = [0 for _ in range(sentence_num)]
    # word frequency statistics
    for i in range(sentence_num):
        words = lines[i].split(' ')
        sentence_lens[i] = len(words)  # record the number of words in a sentence
        if sentence_lens[i] == 0:
            sentence_lens[i] = 1  # avoid division by zero
        for word in words:
            if word not in index:
                index[word] = dict()
            if i not in index[word]:
                index[word][i] = 1
            else:
                index[word][i] += 1

    # convert to tf-idf matrix
    for word in index:
        sen_cov_word = len(index[word])
        idf = math.log(sentence_num / sen_cov_word, 10) + 1
        for sen_id in index[word]:
            tf = index[word][sen_id] / sentence_lens[sen_id]
            index[word][sen_id] = tf * idf

    logging.info("finish index generating")
    return index, sentence_num


def cal_similarity(index: dict, sen_num: int) -> dict:
    sen_len = [0 for _ in range(sen_num)]
    simi_matrix = list()
    for i in range(sen_num):
        simi_matrix.append([0 for _ in range(i + 1)])

    # go through and compute all the multiplications
    for word in index:
        ids = [int(i) for i in index[word].keys()]
        for sen_id in ids:
            # calculate length^2 of sentences
            sen_len[sen_id] += index[word][sen_id] ** 2

        for i in range(len(ids)):
            for j in range(i, len(ids)):
                simi_matrix[int(ids[j])][int(ids[i])] += index[word][ids[j]] * index[word][ids[i]]

    # get the length of sentences
    for i in range(sen_num):
        sen_len[i] = math.sqrt(sen_len[i])

    # normalize results
    for i in range(sen_num):
        for j in range(i + 1):
            simi_matrix[i][j] /= sen_len[i] * sen_len[j]

    return simi_matrix


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    start = time.time()
    index, sen_num = __gene_index("data/data.txt")
    result = cal_similarity(index, sen_num)
    with open('data/result.txt', 'w', encoding='utf-8') as out:
        for r in result:
            out.write(str(r) + "\n")
    end = time.time()
    logging.info(f"time cost: {end - start} s")
