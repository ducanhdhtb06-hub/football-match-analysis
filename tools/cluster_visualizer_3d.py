import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import supervision as sv

from src.config import ModelConfig, ClassIDConfig, PipelineConfig
from src.detection import PlayerDetector
from src.team_classifier import TeamClassifier, collect_player_crops


def pil_image_to_data_uri(image: Image.Image) -> str:
    """Encode PIL image to base64 data URI."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def export_interactive_3d_html(
    labels: np.ndarray,
    projections: np.ndarray,
    crops: List[np.ndarray],
    output_html_path: str,
    show_legend: bool = True,
) -> None:
    """
    Generate an interactive 3D HTML visualization using Plotly with embedded clickable image previews.
    (Modularized from Cell 20 of the notebook).
    """
    import plotly.graph_objects as go

    pil_images = [sv.cv2_to_pillow(c) for c in crops]
    image_data_uris = {f"image_{i}": pil_image_to_data_uri(img) for i, img in enumerate(pil_images)}
    image_ids = np.array([f"image_{i}" for i in range(len(pil_images))])

    unique_labels = np.unique(labels)
    traces = []
    colors = ["#00BFFF", "#FF1493", "#FFD700", "#32CD32"]

    for idx, unique_label in enumerate(unique_labels):
        mask = labels == unique_label
        customdata_masked = image_ids[mask]
        trace = go.Scatter3d(
            x=projections[mask][:, 0],
            y=projections[mask][:, 1],
            z=projections[mask][:, 2],
            mode="markers",
            text=labels[mask],
            customdata=customdata_masked,
            name=f"Team {unique_label}",
            marker=dict(
                size=7,
                color=colors[idx % len(colors)],
                opacity=0.85,
            ),
            hovertemplate="<b>Team: %{text}</b><br>ID: %{customdata}<extra></extra>",
        )
        traces.append(trace)

    min_val = np.min(projections)
    max_val = np.max(projections)
    padding = (max_val - min_val) * 0.05
    axis_range = [min_val - padding, max_val + padding]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Player Jersey Color Embedding Clusters (SigLIP + UMAP/PCA + KMeans)",
        scene=dict(
            xaxis=dict(title="X", range=axis_range),
            yaxis=dict(title="Y", range=axis_range),
            zaxis=dict(title="Z", range=axis_range),
            aspectmode="cube",
        ),
        width=1100,
        height=850,
        showlegend=show_legend,
    )

    plotly_div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="scatter-plot-3d")

    javascript_code = f"""
    <script>
        var imageDataURIs = {image_data_uris};
        function displayImage(imageId) {{
            var imageElement = document.getElementById('image-display');
            var placeholderText = document.getElementById('placeholder-text');
            if (imageDataURIs[imageId]) {{
                imageElement.src = imageDataURIs[imageId];
                imageElement.style.display = 'block';
                placeholderText.style.display = 'none';
            }}
        }}

        var chartElement = document.getElementById('scatter-plot-3d');
        if (chartElement) {{
            chartElement.on('plotly_click', function(data) {{
                if (data.points && data.points.length > 0) {{
                    var customdata = data.points[0].customdata;
                    displayImage(customdata);
                }}
            }});
        }}
    </script>
    """

    html_template = f"""<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>Football Player Clustering 3D View</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f8f9fa;
            }}
            #container {{
                position: relative;
                display: flex;
            }}
            #image-container {{
                position: fixed;
                top: 40px;
                right: 40px;
                width: 220px;
                height: 280px;
                padding: 10px;
                border: 2px solid #ccc;
                border-radius: 8px;
                background-color: white;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 1000;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }}
            #image-display {{
                max-width: 100%;
                max-height: 220px;
                object-fit: contain;
                border-radius: 4px;
            }}
            #placeholder-text {{
                color: #888;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <h2>3D Interactive Player Clustering</h2>
        <p>Click on any marker in the 3D plot to inspect the corresponding player crop image.</p>
        <div id="container">
            {plotly_div}
            <div id="image-container">
                <img id="image-display" src="" alt="Selected player crop" style="display: none;" />
                <p id="placeholder-text">👉 Click a 3D data point to preview player crop</p>
            </div>
        </div>
        {javascript_code}
    </body>
</html>
"""
    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"[Success] Interactive 3D visualization saved to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Extract player crops and create 3D interactive clustering report.")
    parser.add_argument("--video", type=str, default="football/test.mp4", help="Path to input football video")
    parser.add_argument("--weights", type=str, default=None, help="YOLO model weights path")
    parser.add_argument("--stride", type=int, default=20, help="Frame sampling stride")
    parser.add_argument("--max-crops", type=int, default=150, help="Max player crops to extract")
    parser.add_argument("--output", type=str, default="reports/player_clusters_3d.html", help="Path to output HTML")
    args = parser.parse_args()

    model_config = ModelConfig()
    if args.weights:
        model_config.yolo_weights_path = args.weights

    detector = PlayerDetector(model_config=model_config)
    crops = collect_player_crops(
        video_path=args.video,
        detector=detector,
        stride=args.stride,
        max_crops=args.max_crops,
    )

    if len(crops) < 2:
        print(f"Error: Only collected {len(crops)} crops. Need at least 2.")
        return

    classifier = TeamClassifier(device=model_config.device)
    classifier.fit(crops)
    projections = classifier.get_projections(crops)
    labels = classifier.predict(crops)

    export_interactive_3d_html(
        labels=labels,
        projections=projections,
        crops=crops,
        output_html_path=args.output,
    )


if __name__ == "__main__":
    main()
