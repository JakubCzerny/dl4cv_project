import xml.etree.cElementTree as ET

def load_annotations(fname, target_size):
    bboxes = {}
    original_size = {}
    tree = ET.ElementTree(file='annotations/'+fname)
    to_be_extracted = ['xmin','ymin','xmax','ymax']

    for tag in ['width', 'height']:
        for elem in tree.iter(tag=tag):
            original_size[tag] = int(elem.text)

    for tag in to_be_extracted:
        for idx,elem in enumerate(tree.iter(tag=tag)):
            value = float(elem.text)
            idx = str(idx)

            if bboxes.has_key(idx):
                bboxes[idx].append(value)
            else:
                bboxes[idx] = [value]

    width_ratio  = float(target_size[0]) / original_size['width']
    height_ratio = float(target_size[1]) / original_size['height']

    print width_ratio, height_ratio

    for bbox in bboxes.values():
        print bbox[0]
        bbox[0] *= width_ratio
        bbox[1] *= height_ratio
        bbox[2] *= width_ratio
        bbox[3] *= height_ratio

    x0 = bboxes['0'][0]
    y0 = bboxes['0'][1]
    w  = bboxes['0'][2] - bboxes['0'][0]
    h  = bboxes['0'][3] - bboxes['0'][1]
    return [x0,y0,w,h]
