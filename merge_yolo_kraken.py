import time
import numpy as np
import json
import skimage
from kraken import blla
from kraken.lib import models
from kraken import rpred
from kraken.lib import vgsl
from kraken import transcribe as transcribe
import glob
from ultralytics import YOLO as YOLO
from PIL import Image
import cv2
import matplotlib.pyplot as plt 



def write_json(path:str, object:json) -> None:
    with open(path, "w") as output_file:
        json.dump(object, output_file)


def read_json(path:str) -> dict:
    with open(path, "r") as output_file:
        mydict = json.load(output_file)
    return mydict


class Document:
    def __init__(self, path, extension, yolo_model_path, kraken_segmentation_model_path, kraken_transcription_model):
        self.fragments = dict()
        self.path = path
        self.extension = extension
        self.segmented_fragments = list()
        self.annotated_fragments = list()
        self.kraken_transcription_model = kraken_transcription_model
        self.yolo_model = YOLO(yolo_model_path)
        self.kraken_segmentation_model = vgsl.TorchVGSLModel.load_model(kraken_segmentation_model_path)
    
    def run_retrieval(self):
        print("Starting process")
        print(f"{self.path}/*.{self.extension}")
        for page in glob.glob(f"{self.path}/pg_02*{self.extension}"):
            print(page)
            NewPage = Page(page, self.yolo_model, self.kraken_segmentation_model, self.kraken_transcription_model)
            NewPage.get_annotated_lines()
            if NewPage.annotated_lines:
                print("Segmenting")
                NewPage.segment_lines()
                NewPage.get_lines()
                NewPage.transcribe()
            # zipped = list(zip(NewPage.annotated_fragments, NewPage.segmented_fragments, NewPage.transcribed_fragments))
            # self.fragments[page] = zipped
            
class Page():
    def __init__(self, page, yolo_model, kraken_segmentation_model, kraken_transcription_model):
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
        self.transcription_model = models.load_any(kraken_transcription_model, device="cuda:0")
    
    
    def get_annotated_lines(self):
        """
        Writes to a specific folder
        :param yolo_model: 
        :return: 
        """
        # Here happens the yolo magick
        confidence_threshold = .25
        self.annotated_lines = yolo_run(self.page, self.yolo_model, confidence_threshold)
        if self.annotated_lines:
            write_json("annotations.json", self.annotated_lines.boxes.xyxy.tolist())
            extract_annotations(self.page, self.annotated_lines)
    
    def get_lines(self):
        """
        This function takes 
        :param kraken_segmentation: 
        :return: 
        """
        
        for annotation in self.annotated_lines:
            annot = [round(coord) for coord in annotation]
            x_1, y_1, x_2, y_2 = annot
            full_rectangle = [(x_1, y_1), (x_1, y_2), (x_2, y_2), (x_2, y_1)]
            annotated_lines = []
            
            # On va chercher les lignes dans le rectangle identifié
            for index, line in enumerate(self.segmentation['lines']):
                baseline = line['baseline']
                bx_1 = baseline[0][0]
                by_1 = baseline[0][1]
                bx_2 = baseline[1][0]
                by_2 = baseline[1][1]
                if all([bx_1 > x_1, bx_2 < x_2, by_1 > y_1, by_2 < y_2]):
                    print("Found line")
                    annotated_lines.append(line)
            print(len(annotated_lines))
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
        basename = self.page.replace(".png", f"_{index}.txt")
        with open(basename, "w") as output_txt_file:
            output_txt_file.write("\n".join(self.predictions))
        
    
    def segment_lines(self):
        print(f"Segmenting {self.page}")
        fragment = Image.open(self.page)
        self.segmentation = blla.segment(fragment, model=self.kraken_segmentation_model, device="cuda:0")
        
            
    def transcribe(self):
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
        im_new.save(f"results/extracted_fragments/{basename}_{index}.png")
        print(f'Saved to results/extracted_fragments/{basename}_{index}.png')

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
    target_mss = Document(path="/home/mgl/Bureau/Travail/Reproductions/Aegidius_Romanus/Version_B/Esc_Q/burst", extension='png', 
                          yolo_model_path="train_results/train31/weights/best.pt", 
                          kraken_segmentation_model_path="models/global_model_bl_best.mlmodel", 
                          kraken_transcription_model="models/modele_finetuned_best.mlmodel")
    target_mss.run_retrieval()
    
if __name__ == '__main__':
    main()