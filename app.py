import os
from flask import Flask, request, jsonify
import pandas as pd
from PIL import Image, ImageDraw
import io

app = Flask(__name__)

@app.route("/api/annotate", methods=["POST"])
def annotate_image():
    image_file = request.files.get("image")
    if not image_file:
        return jsonify({"error": "No image uploaded"}), 400

    # Load image
    image = Image.open(image_file.stream).convert("RGB")

    # Load your Excel file (make sure train.xlsx is in the same folder as app.py or use full path)
    df = pd.read_excel("train.xlsx")

    image_id = os.path.splitext(image_file.filename)[0]
    matches = df[df.iloc[:, 0].str.contains(image_id)]

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)
    labels = []
    for _, row in matches.iterrows():
        label = str(row[1])
        xmin, ymin, xmax, ymax = int(row[2]), int(row[3]), int(row[4]), int(row[5])
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 10), label, fill="red")
        labels.append(label)

    # Save to memory
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    return jsonify({
        "labels": list(set(labels)),
        "message": f"Detected {len(labels)} annotations"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Required for Render
    app.run(host="0.0.0.0", port=port)
