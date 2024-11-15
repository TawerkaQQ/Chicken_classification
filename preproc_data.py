import numpy as np
import argparse
import os

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    """
    ann-path - path to annotations
    data-path - path to images
    """

    parser.add_argument('--ann-path', type=Path,
                        default=f'{os.getcwd()}/anot/_annotations.coco.json')
    parser.add_argument('--data-path', type=str,
                        default=f'{os.getcwd()}/train/')
    parser.add_argument('--output-folder', type=Path, required=True)

    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()
    # parser.add_argument()

    return parser.parse_args()


def main():

    args = parse_args()

    man_subfolder = args.output_folder / 'man'
    woman_subfolder = args.output_folder / 'woman'

    man_subfolder.mkdir(parents=True, exist_ok=True)
    woman_subfolder.mkdir(parents=True, exist_ok=True)

    # Initialize the COCO api for instance annotations
    coco = COCO(args.ann_path)
    cat_ids = coco.getCatIds()
    print("Category IDs:", cat_ids)  # The IDs are not necessarily consecutive.

    # All categories.
    cats = coco.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:", cat_names)

    for cat_idx in range(1, len(cat_names)):
        query_name = cat_names[cat_idx]
        query_id = coco.getCatIds(catNms=[query_name])[0]
        print(f"Category Name: {query_name}, Category ID: {query_id}")

        # Get the ID of all the images containing the object of the category.
        img_ids = coco.getImgIds(catIds=[query_id])
        print(f"Number of Images Containing {query_name}: {len(img_ids)}")

        # Retrieve image dimensions
        for img in (tqdm(range(len(img_ids)), unit_scale=True, total=len(img_ids), leave=True,
                    bar_format='Осталось {remaining}')):

            img_id = img_ids[img]
            img_info = coco.loadImgs([img_id])[0]
            image_path = args.data_path + img_info['file_name']

            annotation_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(annotation_ids)

            mask = coco.annToMask(annotations[0])

            # Apply the binary mask to the original image
            image = Image.open(image_path)  # Load the original image
            image_array = np.array(image)  # Convert the image to a NumPy array

            image_array[mask == 0] = 0

            # Save or display the object image
            object_image = Image.fromarray(image_array)

            if cat_idx == 1:
                object_image.save(f"{man_subfolder}/{img_id}.jpg")
            else:
                object_image.save(f"{woman_subfolder}/{img_id}.jpg")


if __name__ == '__main__':
    main()
