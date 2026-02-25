from flask import Flask, render_template, request, jsonify
import base64
import re

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/upload_frame")
def upload_frame():
    data = request.json.get("image", "")
    # verwacht een data URL: "data:image/jpeg;base64,...."
    m = re.match(r"^data:image/\w+;base64,(.+)$", data)
    if not m:
        return jsonify({"ok": False, "error": "Invalid image data"}), 400

    img_b64 = m.group(1)
    img_bytes = base64.b64decode(img_b64)

    # demo: schrijf het laatste frame weg
    with open("last_frame.jpg", "wb") as f:
        f.write(img_bytes)

    return jsonify({"ok": True, "bytes": len(img_bytes)})

if __name__ == "__main__":
  app.run(
      host="0.0.0.0",
      port=5000,
      ssl_context=("localhost+2.pem", "localhost+2-key.pem"),
      debug=True
  )