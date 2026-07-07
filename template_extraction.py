import json
import os
from collections import defaultdict
from PIL import Image

# ============================
# CONFIGURAÇÃO
# ============================

JSON_FILE = "data/12-00.json"   # <-- Altere para o nome do seu JSON
IMAGE_DIR = "data/12-00"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def has_intersection(rect1, rect2):
    """
    Verifica se dois retângulos possuem interseção.

    rect = {
        'x': ...,
        'y': ...,
        'width': ...,
        'height': ...
    }
    """

    x1_min = rect1["x"]
    y1_min = rect1["y"]
    x1_max = x1_min + rect1["width"]
    y1_max = y1_min + rect1["height"]

    x2_min = rect2["x"]
    y2_min = rect2["y"]
    x2_max = x2_min + rect2["width"]
    y2_max = y2_min + rect2["height"]

    return not (
        x1_max <= x2_min
        or x1_min >= x2_max
        or y1_max <= y2_min
        or y1_min >= y2_max
    )


# ============================
# CARREGA JSON COCO
# ============================

with open(JSON_FILE, "r") as f:
    coco = json.load(f)

# image_id -> nome da imagem
image_map = {}

for img in coco["images"]:
    image_map[img["id"]] = os.path.basename(img["file_name"])

# agrupa anotações por imagem
annotations_by_image = defaultdict(list)

for ann in coco["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)

print(f"{len(image_map)} imagens encontradas.")
print(f"{len(coco['annotations'])} anotações encontradas.\n")


# ============================
# PROCESSAMENTO
# ============================

saved = 0
skipped = 0

for image_id, annotations in annotations_by_image.items():

    filename = image_map[image_id]
    image_path = os.path.join(IMAGE_DIR, filename)

    if not os.path.exists(image_path):
        print(f"[AVISO] Imagem não encontrada: {image_path}")
        continue

    conflict_regions = set()

    # Detecta sobreposição
    for i in range(len(annotations)):
        x1, y1, w1, h1 = annotations[i]["bbox"]

        rect1 = {
            "x": x1,
            "y": y1,
            "width": w1,
            "height": h1,
        }

        for j in range(i + 1, len(annotations)):
            x2, y2, w2, h2 = annotations[j]["bbox"]

            rect2 = {
                "x": x2,
                "y": y2,
                "width": w2,
                "height": h2,
            }

            if has_intersection(rect1, rect2):
                conflict_regions.add(i)
                conflict_regions.add(j)

    with Image.open(image_path) as img:

        for idx, ann in enumerate(annotations):

            if idx in conflict_regions:
                skipped += 1
                continue

            x, y, w, h = ann["bbox"]

            cropped = img.crop(
                (
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h),
                )
            )

            output_name = (
                f"{os.path.splitext(filename)[0]}"
                f"_region_{ann['id']}.jpg"
            )

            cropped.save(
                os.path.join(
                    OUTPUT_DIR,
                    output_name,
                )
            )

            saved += 1

print("\n===================================")
print("Extração concluída.")
print(f"Templates salvos : {saved}")
print(f"Regiões ignoradas: {skipped}")
print("===================================")