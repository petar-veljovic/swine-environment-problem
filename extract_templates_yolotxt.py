import os
from PIL import Image


def has_intersection(rect1, rect2):
    """Verifica se dois retângulos possuem interseção."""

    x1_min = rect1["x"]
    y1_min = rect1["y"]
    x1_max = rect1["x"] + rect1["width"]
    y1_max = rect1["y"] + rect1["height"]

    x2_min = rect2["x"]
    y2_min = rect2["y"]
    x2_max = rect2["x"] + rect2["width"]
    y2_max = rect2["y"] + rect2["height"]

    return not (
        x1_max <= x2_min or
        x1_min >= x2_max or
        y1_max <= y2_min or
        y1_min >= y2_max
    )


# Pasta contendo imagens e labels
image_dir = "images"
label_dir = "labels"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

# Percorre todas as imagens
for image_name in os.listdir(image_dir):

    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_dir, image_name)
    label_path = os.path.join(
        label_dir,
        os.path.splitext(image_name)[0] + ".txt"
    )

    if not os.path.exists(label_path):
        print(f"Label não encontrado: {label_path}")
        continue

    with Image.open(image_path) as img:

        img_width, img_height = img.size

        rectangles = []

        # Lê arquivo YOLO
        with open(label_path, "r") as f:
            for idx, line in enumerate(f):

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls, xc, yc, w, h = map(float, parts)

                # Converte YOLO -> pixels
                box_w = w * img_width
                box_h = h * img_height

                x = (xc * img_width) - box_w / 2
                y = (yc * img_height) - box_h / 2

                rectangles.append({
                    "id": idx,
                    "class": int(cls),
                    "x": x,
                    "y": y,
                    "width": box_w,
                    "height": box_h
                })

        conflict_regions = set()

        # Detecta sobreposição
        for i in range(len(rectangles)):
            for j in range(i + 1, len(rectangles)):

                if has_intersection(rectangles[i], rectangles[j]):
                    conflict_regions.add(i)
                    conflict_regions.add(j)

        # Salva apenas caixas sem conflito
        for i, rect in enumerate(rectangles):

            if i in conflict_regions:
                continue

            left = int(max(0, rect["x"]))
            top = int(max(0, rect["y"]))
            right = int(min(img_width, rect["x"] + rect["width"]))
            bottom = int(min(img_height, rect["y"] + rect["height"]))

            cropped = img.crop((left, top, right, bottom))

            output_name = (
                f"{os.path.splitext(image_name)[0]}"
                f"_region_{i}.jpg"
            )

            cropped.save(os.path.join(output_dir, output_name))

print("Extração concluída. Todas as regiões sobrepostas foram ignoradas.")