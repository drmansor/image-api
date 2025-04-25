from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# Load the Excel data once
EXCEL_PATH = os.environ.get("EXCEL_PATH", "train.xlsx")
excel_data = pd.read_excel(EXCEL_PATH)

labels_list = ["Aortic", "enlargement", "Cardiomegaly", "Pleural", "thickening",
               "Pulmonary", "fibrosis", "Covid", "Pneumonia"]

label_colors = {
    "Aortic enlargement": (255, 0, 0),
    "Cardiomegaly": (0, 0, 255),
    "Pleural thickening": (0, 128, 0),
    "Pulmonary fibrosis": (255, 165, 0),
    "Covid-19": (255, 242, 0),
    "Pneumonia": (0, 255, 255)
}

def read_detections(image_id, img_height):
    detections = []
    matched_rows = excel_data[excel_data.iloc[:, 0].astype(str).str.contains(image_id, na=False)]
    for _, row in matched_rows.iterrows():
        label = str(row[1]).strip()
        xmin = int(row[2])
        ymin = int(row[3])
        xmax = int(row[4])
        ymax = int(row[5])
        detections.append({
            "label": label,
            "x": xmin,
            "y": ymin,
            "width": xmax - xmin,
            "height": ymax - ymin
        })

    if not detections:
        for keyword in labels_list:
            if keyword.lower() in image_id.lower():
                label = {
                    "Aortic": "Aortic enlargement",
                    "enlargement": "Aortic enlargement",
                    "Pleural": "Pleural thickening",
                    "thickening": "Pleural thickening",
                    "Pulmonary": "Pulmonary fibrosis",
                    "fibrosis": "Pulmonary fibrosis",
                    "Covid": "Covid-19"
                }.get(keyword, keyword)
                detections.append({
                    "label": label,
                    "x": int(img_height / 2 + 50),
                    "y": int(img_height / 2 + 100),
                    "width": 100,
                    "height": 150
                })
    return detections

def draw_boxes(image, detections):
    draw = ImageDraw.Draw(image)
    for det in detections:
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        label = det["label"]
        color = label_colors.get(label, (128, 128, 128))
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((x + 5, y - 10), label, fill=color)
    return image

@app.route("/", methods=["GET"])
def home():
    return "🩺 X-ray Annotation API is running"

@app.route("/api/image/annotate", methods=["POST"])
def annotate_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded."}), 400

    image = Image.open(file.stream).convert("RGB")
    image_id = os.path.splitext(file.filename)[0]
    detections = read_detections(image_id, image.height)
    draw_boxes(image, detections)

    # Create report
    report = list(set(det["label"] for det in detections))

    # Encode image as base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "image": img_str,
        "report": report
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
