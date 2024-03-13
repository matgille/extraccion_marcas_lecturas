import glob



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
    frequencies_list = [(item, freq/length_annotated_corpus) for item, freq in frequencies_dict.items()]
    frequencies_list.sort(key=lambda x:x[1])
    return frequencies_list
    
def extract_from_full_text():
    directory = "/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/data/textos/lemmatized"
    with open(f"{directory}/incunable_lemmatized.txt", "r") as lemmatized_file:
        analyses = lemmatized_file.read().split("\n")
    lemmes = [analyse.split()[1] for analyse in analyses if len(analyse.split()) > 0]
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
    frequencies_list = [(item, freq / length_annotated_corpus) for item, freq in frequencies_dict.items()]
    frequencies_list.sort(key=lambda x: x[1])
    print(frequencies_list)
    return frequencies_list


def merge_lists(target_list, compare_list):
    out_list = []
    compare_dict = {item:frequency for item, frequency in compare_list}
    for item, frequency in target_list:
        try:
            if frequency > compare_dict[item]:
                interm_tuple = (item, frequency, compare_dict[item], round((frequency/compare_dict[item]), 2) , "Higher")
            else:
                interm_tuple = (item, frequency, compare_dict[item], -round((frequency/compare_dict[item]), 2) ,"Lower")
            out_list.append(interm_tuple)
        except KeyError:
            continue
    out_list.sort(key=lambda x:x[3])
    print(out_list)
    with open("/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/data/stats/compared_frequencies.csv", "w") as output_tsv:
        output_tsv.write("Lemma\tFreq. in annotations\tFreq. in full text\tDifference\tComparison\n")
        for item in out_list:
            output_tsv.write("\t".join([str(val) for val in item]) + "\n")
            
if __name__ == '__main__':
    annotations = extract_from_annotations()
    full_text = extract_from_full_text()
    merge_lists(annotations, full_text)