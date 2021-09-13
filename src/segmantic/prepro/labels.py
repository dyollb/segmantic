import numpy as np
import colorsys


def build_map(input_label_names, mapper):
    input_label_map = {n: i for i, n in enumerate(input_label_names)}

    output_label_names = list(sorted(set([mapper(n) for n in input_label_names])))
    output_label_names.remove("Background")
    output_label_names = ["Background"] + output_label_names
    output_label_map = {n: i for i, n in enumerate(output_label_names)}

    input2output = np.zeros((len(input_label_names),), dtype=np.uint16)
    for name in input_label_map.keys():
        index = input_label_map[name]
        name_mapped = mapper(name)
        index_mapped = output_label_map[name_mapped]
        input2output[index] = index_mapped
    return input_label_map, output_label_map, input2output


def save_tissue_list(output_label_map:dict, tissue_list_file_name:str):
    with open(tissue_list_file_name, "w") as f:
        num_tissues = max(output_label_map.values())
        label_tissue_map = {}
        for n in output_label_map.keys():
            label_tissue_map[output_label_map[n]] = n

        def random_color(l, max_label):
            if l == 0:
                return (0, 0, 0)
            hue = l / (2.0 * max_label) + (l % 2) * 0.5
            hue = min(hue, 1.0)
            return colorsys.hls_to_rgb(hue, 0.5, 1.0)

        print("V7", file=f)
        print("N%d" % num_tissues, file=f)
        for label in range(1, num_tissues+1):
            r,g,b = random_color(label, num_tissues)
            print("C%.2f %.2f %.2f %.2f %s" % (r,g,b, 0.5, label_tissue_map[label]), file=f)


def load_tissue_list(file_name) -> dict:
    tissues = { 0: "Background" }
    with open(file_name) as f:
        for line in f.readlines():
            if line.startswith('C'):
                tissue = line.rsplit(" ", 1)[-1]
                tissues[len(tissues)] = tissue
    return tissues

