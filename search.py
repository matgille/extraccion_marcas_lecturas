import glob
import json
import sys


def search(dictionnary, search_term):
    print(search_term)
    files = dictionnary[search_term]
    print(files)
    for file in files:
        corresponding_file = file.replace(".lemmatized", "")
        print(f"\nFile: {corresponding_file}")
        with open(corresponding_file, "r") as input_file:
            as_string = input_file.read()
        print(as_string)
    

def index_lemmas():
    files_dict = {}
    for file in glob.glob("results/kraken_transcription_results/*lemmatized"):
        with open(file, "r") as input_file:
            as_list = input_file.read().split("\n")
            as_list = [item for item in as_list if item != ""]
            lemmas = [item.split(" ")[1] for item in as_list]
            files_dict[file] = lemmas
        
    lemmas_dict = {}
    for file, lemmas in files_dict.items():
        for lemma in lemmas:
            try:
                if file not in lemmas_dict[lemma]:
                    lemmas_dict[lemma].append(file)
            except KeyError:
                lemmas_dict[lemma] = [file]
    return lemmas_dict


if __name__ == '__main__':
    lemmas_dict = index_lemmas()
    search_term = sys.argv[1]
    search(lemmas_dict, search_term)