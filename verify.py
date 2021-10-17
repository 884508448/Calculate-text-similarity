import re
from heapq import *


def get_most_alike(index, num):
    heap = [0 for i in range(num)]
    heapify(heap)
    index_arr = [0 for i in range(num)]
    simi_arr = None
    with open("data/imp_para_result.txt", encoding='utf-8') as file:
        for id, line in enumerate(file):
            if id >= index:
                line = line[1:-2].split(",")
                if id == index:
                    simi_arr = [float(it) for it in line]
                else:
                    simi_arr.append(float(line[index]))
    for it in simi_arr:
        if it > heap[0]:
            heapreplace(heap, it)
    heap.sort(reverse=True)
    with open("data/data.txt", encoding='utf-8') as file:
        lines = file.readlines()
    print("original:", re.sub(r'\s', '', lines[index]))
    print("match:")
    count = 0
    for st in range(len(simi_arr)):
        if (simi_arr[st] in heap):
            index_arr[heap.index(simi_arr[st])] = st
            count += 1
            if count == num:
                break
    for id in index_arr:
        print(re.sub(r'\s', '', lines[id]))


if __name__ == '__main__':
    get_most_alike(17, 10)
