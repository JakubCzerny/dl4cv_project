import xml.etree.cElementTree as ET

def load_annotations(fname):
    bboxes = {}
    tree = ET.ElementTree(file='annotations/'+fname)
    to_be_extracted = ['xmin','ymax','xmax','ymax']

    for tag in to_be_extracted:
        for idx,elem in enumerate(tree.iter(tag=tag)):
            value = float(elem.text)
            idx = str(idx)

            if bboxes.has_key(idx):
                bboxes[idx].append(value)
            else:
                bboxes[idx] = [value]

    return bboxes['0']
