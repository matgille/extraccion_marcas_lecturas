import os
import sys
import numpy as np
import json
from kraken import blla
from kraken.lib import models
from kraken import rpred
from kraken.lib import vgsl
import glob
from ultralytics import YOLO as YOLO
from PIL import Image
import unicodedata



def write_json(path:str, object:json) -> None:
    with open(path, "w") as output_file:
        json.dump(object, output_file)


def read_json(path:str) -> dict:
    with open(path, "r") as output_file:
        mydict = json.load(output_file)
    return mydict


class Document:
    def __init__(self, 
                 path, 
                 extension, 
                 yolo_model_path, 
                 kraken_segmentation_model_path, 
                 kraken_transcription_model,
                 overwrite_extraction,
                 overwrite_segmentation,
                 overwrite_transcription,
                 overwrite_normalization):
        self.fragments = dict()
        self.path = path
        self.extension = extension
        self.segmented_fragments = list()
        self.annotated_fragments = list()
        self.kraken_transcription_model = kraken_transcription_model
        self.yolo_model = YOLO(yolo_model_path)
        self.kraken_segmentation_model = vgsl.TorchVGSLModel.load_model(kraken_segmentation_model_path)
        self.kraken_transcription_model = models.load_any(kraken_transcription_model, device="cuda:0")
        self.overwrite_extraction = overwrite_extraction
        self.overwrite_segmentation = overwrite_segmentation
        self.ovewrite_transcription = overwrite_transcription
        self.overwrite_normalization = overwrite_normalization
        # Let's create the dirs
        for directory in ["results",
                          "results/yolo_extracted_fragments/", 
                          "results/kraken_segmentation_results", 
                          "results/kraken_transcription_results"]:
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass
    
    def run_retrieval(self):
        print("Starting process")
        print(f"{self.path}/*.{self.extension}")
        for page in glob.glob(f"{self.path}/pg_0*{self.extension}"):
            print(page)
            NewPage = Page(page, 
                           self.yolo_model, 
                           self.kraken_segmentation_model, 
                           self.kraken_transcription_model)
            NewPage.get_annotated_lines(overwrite_extraction=self.overwrite_extraction)
            if NewPage.annotated_lines:
                print("Segmenting")
                NewPage.segment_lines(overwrite_segmentation=self.overwrite_segmentation)
                NewPage.get_lines()
                NewPage.transcribe(overwrite_transcription=self.ovewrite_transcription)
                NewPage.normalize(overwrite_normalization=self.overwrite_normalization)
            
class Page():
    def __init__(self, page, yolo_model, kraken_segmentation_model, kraken_transcription_model):
        self.basename = page.split("/")[-1].split(".")[0]
        self.annotated_fragments = list()
        self.fragments = dict()
        self.page = page
        self.segmented_fragments = list()
        self.transcribed_fragments = list()
        self.yolo_model = yolo_model
        self.kraken_segmentation_model = kraken_segmentation_model
        self.kraken_transcription_model = kraken_transcription_model
        self.segmentation = None
        self.annotated_lines = None
        self.image = Image.open(page)
        self.annotated_fragments_in_page = []
        self.transcription_model = kraken_transcription_model
    
    
    def normalize(self, overwrite_normalization):
        with open("models/abreviation_table.tsv", "r") as abreviation_table:
            table = [item.replace("\n", "").split("\t") for item in abreviation_table.readlines()]
        for file in glob.glob(f"results/kraken_transcription_results/{self.basename}*.txt"):
            if os.path.isfile(file) and not overwrite_normalization:
                print("Normalization file exists, passing")
            else:
                with open(file, "r") as transcription:
                    transcription_as_string = transcription.read()
                    print("Orig text:")
                print(transcription_as_string)
                for orig, reg in table:
                    transcription_as_string = transcription_as_string.replace(orig, reg)
                print("Normalized text:")
                print(transcription_as_string)
                with open(file.replace('.txt', '.normalized.txt'), "w") as normalized_file:
                    correctly_encoded = unicodedata.normalize("NFC", transcription_as_string)
                    normalized_file.write(correctly_encoded)
            
    def get_annotated_lines(self, overwrite_extraction):
        """
        Writes to a specific folder
        :param yolo_model: 
        :return: 
        """
        # Here happens the yolo magick
        confidence_threshold = .25
        if os.path.isfile(f"results/yolo_extracted_fragments/{self.basename}.json") and not overwrite_extraction:
            print("Annotations already identified. Passing.")
            self.annotated_lines = read_json(f"results/yolo_extracted_fragments/{self.basename}.json")
        else:
            self.annotated_lines = yolo_run(self.page, self.yolo_model, confidence_threshold)
            if self.annotated_lines:
                write_json(f"results/yolo_extracted_fragments/{self.basename}.json", self.annotated_lines.boxes.xyxy.tolist())
                extract_annotations(self.page, self.annotated_lines)
    
    def get_lines(self):
        """
        This function extract the annotated lines from the kraken segmented page
        :param kraken_segmentation: 
        :return: 
        """
        
        for annotation in self.annotated_lines:
            try:
                annotation_as_list = annotation.boxes.xyxy.tolist()[0]
            except AttributeError:
                annotation_as_list = annotation
            annot = [round(coord) for coord in annotation_as_list]
            x_1, y_1, x_2, y_2 = annot
            annotated_lines = []
            # On va chercher les lignes dans le rectangle identifié
            for index, line in enumerate(self.segmentation['lines']):
                baseline = line['baseline']
                bx_1 = baseline[0][0]
                by_1 = baseline[0][1]
                bx_2 = baseline[1][0]
                by_2 = baseline[1][1]
                if all([bx_1 > x_1, bx_2 < x_2, by_1 > y_1, by_2 < y_2]):
                    annotated_lines.append(line)
            new_dict = {"text-direction": "horizontal-lr", 
                        "type": "baselines", 
                        "lines": annotated_lines,
                        "regions": "",
                        "script_detection": True}
            self.annotated_fragments_in_page.append(new_dict)
        
        
    def predict(self, index, lines_to_predict:dict):
        pred_it = rpred.rpred(self.transcription_model, self.image, lines_to_predict)
        self.predictions = []
        for record in pred_it:
            self.predictions.append(record.prediction)
        basename = self.page.split("/")[-1].replace(".png", f"_{index}.txt")
        with open(f"results/kraken_transcription_results/{basename}", "w") as output_txt_file:
            output_txt_file.write("\n".join(self.predictions))
        
    
    def segment_lines(self, overwrite_segmentation):
        target_file = f"results/kraken_segmentation_results/{self.basename}.json"
        if os.path.isfile(target_file) and not overwrite_segmentation:
            print("Page already segmented. Passing")
            self.segmentation = read_json(target_file)
        else:
            print(f"Segmenting {self.page}")
            fragment = Image.open(self.page)
            self.segmentation = blla.segment(fragment, model=self.kraken_segmentation_model, device="cuda:0")
            write_json(target_file, self.segmentation)
        
            
    def transcribe(self, overwrite_transcription):
        target_files = glob.glob(f"results/kraken_transcription_results/{self.basename}*.txt")
        if len(target_files) > 0 and not overwrite_transcription:
            print("Existing transcription, passing")
        else:
            print("Transcribing annotated fragments")
            for index, fragment in enumerate(self.annotated_fragments_in_page):
                self.predict(index, fragment)
            


def add_margin(pil_img, top, right, bottom, left, color):
    pil_img = Image.fromarray(pil_img)
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result





def extract_annotations(page, yolo_result):
    basename = page.split("/")[-1].split(".")[0]
    coords_as_tensors = yolo_result.boxes.xyxy
    print(coords_as_tensors)
    list_of_coords = coords_as_tensors.tolist()
    for index, labels in enumerate(list_of_coords):
        labels = [int(coord) for coord in labels]
        # https://stackoverflow.com/a/61099508
        # img = cv2.imread(page)
        img = Image.open(page)
        img = np.array(img)

        x_1, y_1, x_2, y_2 = labels
        height = img.shape[0]
        width = img.shape[1]
        segmented_img = img[y_1:y_2, x_1:x_2, :]
        im_new = add_margin(segmented_img, 500, 200, 500, 200, (205,196,175))
        im_new.save(f"results/kraken_transcription_results/{basename}_{index}.png")
        print(f'Saved to results/kraken_transcription_results/{basename}_{index}.png')

def yolo_run(page, model, confidence_threshold):
    # On ne choisit que la classe "lignes horizontales"
    inversed_names = {name:index for index, name in model.names.items()}
    classes = inversed_names['Horizontal-lines']
    # from PIL
    im1 = Image.open(page)
    preds = model.predict(source=im1, classes=classes, conf=confidence_threshold, device='cuda:0')
    if len(preds[0]) == 0:
        print("Nothing found")
        results = None
    else:
        print("Found a box")
        results = preds[0]
    return results


def kraken_transcribe(fragment, model):
    kraken_transcription = "To be done"
    return kraken_transcription


def main():
    path_to_images = sys.argv[1]
    target_mss = Document(path=path_to_images, 
                          extension='png', 
                          yolo_model_path="train_results/train31/weights/best.pt", 
                          kraken_segmentation_model_path="models/segmentation_bl_v3.mlmodel", 
                          kraken_transcription_model="models/transcription_q_v2.mlmodel",
                          overwrite_extraction=False,
                          overwrite_segmentation=False,
                          overwrite_transcription=True,
                          overwrite_normalization=True)
    target_mss.run_retrieval()
    
if __name__ == '__main__':
    main()