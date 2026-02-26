import argparse
import csv
import json
from html import escape
from pathlib import Path


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def resolve_csv_path(base_dir: Path, record_set: str) -> Path:
    candidate = Path(record_set)
    if candidate.is_file():
        return candidate

    csv_path = base_dir / "records" / record_set / "inference_log.csv"
    if csv_path.is_file():
        return csv_path

    raise FileNotFoundError(
        f"Kon inference_log.csv niet vinden voor '{record_set}'. "
        f"Verwacht pad: {csv_path}"
    )


def load_points(csv_path: Path):
    points = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        normalized_header = [h.strip() for h in header]
        has_frame_id_header = "frame_id" in normalized_header

        for row in reader:
            if not row:
                continue

            values = [v.strip() for v in row]

            # Supported formats:
            # 1) Filename,datetime,frame_id,longitude,latitude,heading,MODEL_PATH,lastlatency
            # 2) Filename,datetime,longitude,latitude,heading,MODEL_PATH,lastlatency
            if has_frame_id_header:
                if len(values) < 8:
                    continue
                filename, dt, frame_id, lon_s, lat_s, heading, model_path, latency = values[:8]
            else:
                if len(values) >= 8:
                    filename, dt, frame_id, lon_s, lat_s, heading, model_path, latency = values[:8]
                elif len(values) >= 7:
                    filename, dt, lon_s, lat_s, heading, model_path, latency = values[:7]
                    frame_id = ""
                else:
                    continue

            lat = parse_float(lat_s)
            lon = parse_float(lon_s)
            if lat is None or lon is None:
                continue

            point = {
                "lat": lat,
                "lon": lon,
                "filename": filename,
                "frame_id": frame_id,
                "heading": heading,
                "model_path": model_path,
                "lastlatency": latency,
                "datetime": dt,
            }
            points.append(point)
    return points


def build_html(points, title):
    center_lat = sum(p["lat"] for p in points) / len(points)
    center_lon = sum(p["lon"] for p in points) / len(points)
    points_json = json.dumps(points, ensure_ascii=True)
    safe_title = escape(title)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{safe_title}</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    .popup-img {{ max-width: 280px; max-height: 220px; display: block; margin-top: 8px; }}
    .meta {{ font: 13px/1.4 Arial, sans-serif; min-width: 220px; }}
    .meta b {{ display: inline-block; width: 84px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const points = {points_json};
    const map = L.map("map").setView([{center_lat}, {center_lon}], 17);

    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const track = points.map(p => [p.lat, p.lon]);
    L.polyline(track, {{ color: "#0066ff", weight: 3, opacity: 0.9 }}).addTo(map);

    for (const p of points) {{
      const marker = L.circleMarker([p.lat, p.lon], {{
        radius: 5,
        color: "#e53935",
        fillColor: "#e53935",
        fillOpacity: 0.9,
        weight: 1
      }}).addTo(map);

      const details = [
        `<div class="meta"><div><b>Frame:</b> ${{p.frame_id || "-"}}</div>`,
        `<div><b>Datetime:</b> ${{p.datetime || "-"}}</div>`,
        `<div><b>Latitude:</b> ${{p.lat}}</div>`,
        `<div><b>Longitude:</b> ${{p.lon}}</div>`,
        `<div><b>Heading:</b> ${{p.heading || "-"}}</div>`,
        `<div><b>Latency:</b> ${{p.lastlatency || "-"}}</div>`,
        `<div><b>Model:</b> ${{p.model_path || "-"}}</div>`
      ];

      if (p.filename) {{
        const src = encodeURI(p.filename);
        details.push(`<img class="popup-img" src="${{src}}" alt="${{p.filename}}" />`);
      }}

      details.push("</div>");
      marker.bindPopup(details.join(""));
    }}

    map.fitBounds(track, {{ padding: [24, 24] }});
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Genereer een interactieve kaart met track + klikbare image popups."
    )
    parser.add_argument(
        "record_set",
        help="Naam van de records submap (bijv. laerbeekbos) of direct pad naar inference_log.csv",
    )
    parser.add_argument(
        "--output",
        help="Output HTML pad. Standaard: naast inference_log.csv als track_map.html",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_path = resolve_csv_path(base_dir, args.record_set)
    points = load_points(csv_path)
    if not points:
        raise ValueError(f"Geen geldige GPS-punten gevonden in {csv_path}")

    output_path = (
        Path(args.output).resolve()
        if args.output
        else csv_path.parent / "track_map.html"
    )
    html = build_html(points, f"Track Map - {csv_path.parent.name}")
    output_path.write_text(html, encoding="utf-8")

    print(f"Kaart geschreven naar: {output_path}")
    print(f"Open dit bestand in je browser: {output_path}")


if __name__ == "__main__":
    main()
