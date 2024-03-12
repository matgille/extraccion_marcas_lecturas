import json




with open("data.json", "r") as input_json_data:
    json_dict = json.load(input_json_data)

code_dict = {"Ligne-verticale":"0",
             "Lignes-horizontales": "1",
             "Manicule": "2"}

for item in json_dict:
    url = item['url']
    print(url)
    print(item)
    try:
        labels = item['labels']
        with open(f"out/labels/{url.split('/')[-1].replace('.png', '.txt')}", "w") as out_txt_file:
            for label in labels:
                rectangle_type = label['rectanglelabels'][0]
                rectangle_code = code_dict[rectangle_type]
                label = {it: label[it] for it in label.keys()}
                print(label)
                out_txt_file.write(" ".join([str(element) for element in [rectangle_code, label['x']/100,label['y']/100, label['width']/100,label['height']/100]]) + "\n")
    except KeyError:
        with open(f"out/labels/{url.split('/')[-1].replace('.png', '.txt')}", "w") as out_txt_file:
            print("Passing")
            pass
        