import re


def get_sentences(path: str):
    with open(path, 'r', encoding='gbk') as file:
        lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = re.sub(r"/[a-zA-Z]+|\s+|\n+", ' ', lines[i])
    with open('data/data.txt', 'w', encoding='utf-8') as file:
        for line in lines:
            file.write(line + "\n")


if __name__ == "__main__":
    get_sentences("data/raw_data.txt")
