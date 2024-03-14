import glob
import numpy as np


def extract_from_annotations():
    directory = "results/kraken_transcription_results/"
    full_list = []
    for file in glob.glob(f"{directory}/*.lemmatized"):
        with open(file, "r") as lemmatized_file:
            analyses = lemmatized_file.read().split("\n")
        lemmes = [analyse.split()[1] for analyse in analyses if len(analyse.split()) > 0]
        full_list.extend(lemmes)
    print(full_list)
    length_annotated_corpus = len(full_list)
    print(length_annotated_corpus)
    frequencies_dict = {}
    for item in full_list:
        try:
            frequencies_dict[item] += 1
        except KeyError:
            frequencies_dict[item] = 1
    print(frequencies_dict)
    print(len(frequencies_dict))
    frequencies_list = [(item, freq, freq/length_annotated_corpus) for item, freq in frequencies_dict.items()]
    frequencies_list.sort(key=lambda x:x[1])
    return frequencies_list
    
def extract_from_full_text():
    directory = "/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/data/textos/lemmatized"
    with open(f"{directory}/incunable_lemmatized.txt", "r") as lemmatized_file:
        analyses = lemmatized_file.read().split("\n")
    lemmes = [analyse.split()[1] for analyse in analyses if len(analyse.split()) > 0 and analyse.split()[1] not in ",;:!?."]
    length_annotated_corpus = len(lemmes)
    print(length_annotated_corpus)
    frequencies_dict = {}
    for item in lemmes:
        try:
            frequencies_dict[item] += 1
        except KeyError:
            frequencies_dict[item] = 1
    print(frequencies_dict)
    print(len(frequencies_dict))
    frequencies_list = [(item, freq, freq / length_annotated_corpus) for item, freq in frequencies_dict.items()]
    frequencies_list.sort(key=lambda x: x[1])
    print(frequencies_list)
    return frequencies_list

def difference(val_1, val_2):
    return val_1/val_2

def merge_lists(target_list, compare_list):
    out_list = []
    compare_dict = {item:(abs_frequency, rel_frequency) for item, abs_frequency, rel_frequency in compare_list}
    
    # On ne sélectionne que les lemmes qui apparaissent dans les fragments annotés
    for item, abs_frequency, rel_frequency in target_list:
        try:
            target_rel_frequency = compare_dict[item][1]
            if rel_frequency > target_rel_frequency:
                interm_tuple = (item, abs_frequency, rel_frequency, target_rel_frequency, difference(rel_frequency, target_rel_frequency) , "Higher")
            else:
                interm_tuple = (item, abs_frequency, rel_frequency, target_rel_frequency, difference(rel_frequency, target_rel_frequency) ,"Lower")
            out_list.append(interm_tuple)
        except KeyError:
            continue
    out_list.sort(key=lambda x:x[2], reverse=True)
    print(out_list)
    with open("/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/scripts/results/stats/compared_frequencies.csv", "w") as output_tsv:
        output_tsv.write("Lemma\tAbs. freq. in annotations\tRel. Freq. in annotations\tRel. Freq. in full text\tDifference\tComparison\n")
        for item in out_list:
            output_tsv.write("\t".join([str(val).replace(".", ",") for val in item]) + "\n")

def frequencies_full_text(full_list):
    out_list = []
    compare_dict = {item: (abs_frequency, rel_frequency) for item, abs_frequency, rel_frequency in full_list}
    for item, abs_frequency, rel_frequency in full_list:
        interm_tuple = (item, abs_frequency, rel_frequency)
        out_list.append(interm_tuple)
    out_list.sort(key=lambda x: x[2], reverse=True)
    with open("/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/scripts/results/stats/full_text_frequencies.csv",
              "w") as output_tsv:
        output_tsv.write(
            "Lemma\tAbs. freq. in full text\tRel. Freq. in full text\n")
        for item in out_list:
            output_tsv.write("\t".join([str(val).replace(".", ",") for val in item]) + "\n")

if __name__ == '__main__':
    annotations = extract_from_annotations()
    full_text = extract_from_full_text()
    merge_lists(annotations, full_text)
    frequencies_full_text(full_text)