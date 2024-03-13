import glob
import re

def extract_main(stopwords_list, pos_regex):
    directory = "/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/data/textos/lemmatized"
    with open(f"{directory}/incunable_lemmatized.txt", "r") as lemmatized_file:
        analyses = lemmatized_file.read().split("\n")
    lemmes = [analyse.split()[1] for analyse in analyses if
              len(analyse.split()) > 0 and not re.match(pos_regex, analyse.split()[2]) and not analyse.split()[
                                                                                                   1] in stopwords_list]
    with open(f"{directory}/as_lemmas/incunable_as_lemmas.txt", "w") as output_file:
        output_file.write("\n".join(lemmes))
        

def extract_annotations(stopwords_list, pos_regex):
    files_path = "/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/scripts/results/kraken_transcription_results/*lemmatized"
    files = glob.glob(files_path)
    all_lemmas = []
    print(files)
    for file in files:
        with open(file, "r") as lemmatized_file:
            analyses = lemmatized_file.read().split("\n")
        lemmes = [analyse.split()[1] for analyse in analyses if len(analyse.split()) > 0 and not re.match(pos_regex, analyse.split()[2]) and not analyse.split()[1] in stopwords_list]
        all_lemmas.extend(lemmes)
    print(all_lemmas)
    with open(f"/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/data/textos/lemmatized/as_lemmas/annotations_as_lemmas.txt", "w") as output_file:
        output_file.write("\n".join(all_lemmas))
        
if __name__ == '__main__':
    pos_regex = re.compile(r"D.*|S.*|Z|C.*|F|PP.*")
    stopwords_list = ["2","a","al_a","al_el","alguno","allí","alos_a","alos_el","aquel","así","bien","ca","como","con",
                      "cosa","cuál","cuándo","dar","de","deber","decir","del_de","del_el","dela","dela_el","delas_de",
                      "delas_el","delos_de","donde","e","el","ela_a","ellos","en","enel_el","enel_en","enla_el","enla_en",
                      "estar","este","grande","haber","hacer","hombre","le","lo","mas","mucho","muy","más","ni","no","o",
                      "otro","para","poder","poner","por","porque","que","saber","se","según","sen","ser","si","sin",
                      "sobre","su","tal","te","todo","uno","él", "tan", "ya"]
    extract_main(stopwords_list, pos_regex)
    extract_annotations(stopwords_list, pos_regex)