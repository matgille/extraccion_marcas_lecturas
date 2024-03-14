import lxml.etree as ET
import glob



def full_text_length():
    with open("/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/data/textos/lemmatized/as_lemmas/incunable_as_lemmas.txt") as inc_file:
        incunable = inc_file.read().split("\n")
    print(f"Tokens in full tex: {len(incunable)}")

def annotations_length():
    with open("/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/data/textos/lemmatized/as_lemmas/annotations_as_lemmas.txt") as annotations_file:
        annotations = annotations_file.read().split("\n")
    print(f"Tokens in annotations: {len(annotations)}")

def number_of_transcribed_line():
    alto_namespace = 'http://www.loc.gov/standards/alto/ns-v4#'
    namespace_declaration = {'alto': alto_namespace}
    transcribed_lines = 0
    for xml_file in glob.glob("/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/data/transcription/Q/*.xml"):
        with open(xml_file, "r") as alto_file:
            as_xml = ET.parse(alto_file)
            transcribed_lines += len(as_xml.xpath("//alto:String[@CONTENT != '']", namespaces=namespace_declaration))
    print(f"Number of lines in the train set: {transcribed_lines}")
            
def segmentation_corpus_size():
    corpus_size = len(glob.glob("/home/mgl/Bureau/Travail/Communications_et_articles/toulouse_mars/data/segmentation/baselines/*.xml"))
    print(f"Segmentation corpus size: {corpus_size}")

if __name__ == '__main__':
    full_text_length()
    annotations_length()
    number_of_transcribed_line()
    segmentation_corpus_size()