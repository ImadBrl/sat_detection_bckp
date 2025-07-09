import os
import gc
from PIL import Image
from tqdm import tqdm
import torch
from config.config import CLASSES_FILE, ANNOTATIONS_DIR
from transformers import AutoProcessor, AutoModelForCausalLM


def load_class_map(classes_file):
    custom_class_map = {}
    with open(classes_file, "r") as f:
        for idx, line in enumerate(f):
            label = line.strip()
            custom_class_map[label] = idx
    return custom_class_map


def convert_to_od_format(data):
    return {
        'bboxes': data.get('<OD>', {}).get('bboxes', []),
        'labels': data.get('<OD>', {}).get('labels', [])
    }


def convert_bboxes_to_yolo(bboxes, labels, image_width, image_height, class_map):
    yolo_lines = []
    class_counter = max(class_map.values()) + 1 if class_map else 0

    for box, label in zip(bboxes, labels):
        x1, y1, x2, y2 = box
        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        if label not in class_map:
            class_map[label] = class_counter
            print(f"Nouvelle classe détectée : {label} → ID {class_counter}")
            class_counter += 1

        class_id = class_map[label]
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines, class_map


def generate_annotations(input_dir, output_dir=ANNOTATIONS_DIR, classes_file=CLASSES_FILE):
    os.makedirs(output_dir, exist_ok=True)

    model_id = 'microsoft/Florence-2-large'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    class_map = load_class_map(classes_file)

    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    task_prompt = "<OD>"

    for img_file in tqdm(images, desc="Florence-2 - Annotation des images"):
        try:
            image_path = os.path.join(input_dir, img_file)
            image = Image.open(image_path).convert("RGB")

            inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=1
            )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height)
            )

            result = convert_to_od_format(parsed)
            yolo_lines, class_map = convert_bboxes_to_yolo(
                result['bboxes'], result['labels'], image.width, image.height, class_map
            )

            txt_file = os.path.splitext(img_file)[0] + ".txt"
            with open(os.path.join(output_dir, txt_file), "w") as f:
                f.write("\n".join(yolo_lines))

        except Exception as e:
            print(f"Erreur avec {img_file} : {e}")
        finally:
            del image
            torch.cuda.empty_cache()
            gc.collect()

    # Sauvegarder les classes mises à jour
    with open(os.path.join(output_dir, "classes.txt"), "w") as f:
        for label, idx in sorted(class_map.items(), key=lambda x: x[1]):
            f.write(f"{label}\n")
