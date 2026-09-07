# ⚽ Football Match Analysis & 2D Tactical Pitch Projection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%20%2F%20v11-00FFFF.svg)](https://docs.ultralytics.com/)
[![Supervision](https://img.shields.io/badge/Roboflow-Supervision-purple.svg)](https://supervision.roboflow.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Hệ thống thị giác máy tính (Computer Vision) toàn diện phục vụ phân tích chiến thuật và thống kê chỉ số trận đấu bóng đá tự động từ video phát sóng (broadcast video). Dự án được module hoá từ nguyên mẫu nghiên cứu thành một hệ thống hoàn chỉnh với Command-Line Interface (CLI) mạnh mẽ và Giao diện Web Trực quan (Streamlit Dashboard).

---

## 🌟 Điểm nổi bật & Tính năng chính

1. 🎯 **Object Detection & Tracking**:
   - Nhận diện 4 lớp thực thể: **Quả bóng (Ball)**, **Thủ môn (Goalkeeper)**, **Cầu thủ (Player)**, và **Trọng tài (Referee)** với mô hình YOLOv8 fine-tuned.
   - Theo dõi quỹ đạo thời gian thực (Multi-Object Tracking) bằng **ByteTrack** (`supervision`), loại bỏ nhiễu phát hiện ở cự ly xa qua ngưỡng chiều cao bounding box.

2. 👕 **Tự động phân cụm màu áo 2 đội (Unsupervised Team Clustering)**:
   - Trích xuất đặc trưng hình ảnh sâu (visual embeddings) của áo đấu bằng mô hình Foundation Vision Transformer: **Google SigLIP** (`google/siglip-base-patch16-224`).
   - Giảm số chiều không gian đặc trưng bằng **UMAP (3D)** và phân cụm không giám sát bằng **K-Means (k=2)**. Không cần gán nhãn thủ công màu áo trước trận đấu.

3. 🧤 **Tự động nhận diện đội cho Thủ môn (Goalkeeper Assignment)**:
   - Tự động gán thủ môn về đúng đội hình dựa trên khoảng cách không gian tới trọng tâm toạ độ (centroid) của từng đội trên mặt sân 2D.

4. 📐 **Phát hiện 32 điểm mốc sân (Pitch Keypoints) & Homography 2D**:
   - Nhận diện 32 điểm mốc chuẩn của sân bóng (đường biên, vòng cấm, chấm phạt đền, vòng tròn giữa sân) bằng Roboflow Pitch Detection Model.
   - Tính toán ma trận biến đổi phối cảnh (**Homography Matrix**) chuyển đổi toạ độ pixel camera sang toạ độ thực tế của sân bóng chuẩn (105m x 68m).
   - Cơ chế ổn định ma trận: Bộ lọc làm mịn thời gian (**Exponential Moving Average Smoothing**), chống rung lắc camera và cơ chế chống nhảy khung hình (**Homography Stickiness & Re-baselining**).

5. 🗺️ **Radar Mini-map & Kiểm soát không gian Voronoi (Spatial Control)**:
   - Vẽ bản đồ chiến thuật radar 2D góc nhìn từ trên cao (Bird's-Eye View) thời gian thực.
   - Tích hợp biểu đồ phân vùng kiểm soát **Voronoi** làm mịn gradient bằng hàm hyperbolic tangent (`tanh`) hiển thị trực quan mức độ kiểm soát không gian của 2 đội.

6. 📊 **Bộ đo lường & Thống kê chỉ số chuyên sâu (Match Statistics Engine)**:
   - **Tỉ lệ cầm bóng (Possession %)**: Tính toán theo thời gian thực và chuỗi thời gian liên tục (sliding window).
   - **Phân bố kiểm soát bóng**: Thống kê theo 1/3 sân (Sân nhà / Giữa sân / Sân khách).
   - **Chạm bóng & Giữ bóng**: Đếm số lần chạm bóng và thời gian giữ bóng của từng cầu thủ.
   - **Tốc độ & Bứt tốc (Speed & Sprints)**: Đo tốc độ tức thời và tốc độ tối đa (km/h) qua bộ lọc **Kalman Filter** khử nhiễu toạ độ; phát hiện bứt tốc ($\ge 25\text{ km/h}$).
   - **Quãng đường di chuyển (Distance Covered)**: Tính tổng quãng đường di chuyển (mét) của từng cầu thủ và toàn đội.
   - **Bản đồ nhiệt (Heatmaps) & Vị trí trung bình**: Bản đồ mật độ di chuyển của cầu thủ và trọng tâm đội hình trên sân 2D.
   - **Báo cáo đa định dạng**: Tự động xuất file `match_stats.json`, `match_stats_players.csv`, `match_stats_teams.csv`, và báo cáo đồ hoạ tương tác `match_stats.html`.

7. ⏱️ **Live In-Video Overlay (HUD)**:
   - Hiển thị thanh tỉ lệ kiểm soát bóng động ở đỉnh khung hình video.
   - Hiển thị thẻ tốc độ (km/h) trên đầu mỗi cầu thủ, tự động đổi màu đỏ nổi bật kèm ký hiệu `*` khi cầu thủ đang bứt tốc.

8. 🖥️ **Streamlit Interactive Web Dashboard**:
   - Giao diện web trực quan để tải lên video trận đấu, tuỳ chỉnh các tham số, chạy phân tích với log thời gian thực.
   - **Sẵn sàng ngay lập tức**: Khi vừa mở Dashboard, hệ thống tự động hiển thị ngay kết quả phân tích mẫu (chỉ số kiểm soát bóng 2 đội, biểu đồ tròn Donut & Timeline luỹ kế, bảng thông số từng cầu thủ và thông tin đường dẫn file video đã lưu trên máy).

9. 🔬 **Bộ công cụ chẩn đoán & Visualizer 3D**:
   - Trực quan hoá không gian phân cụm màu áo 3D với Plotly HTML tương tác, nhúng ảnh crop cầu thủ xem trực tiếp khi rê chuột hoặc click vào điểm dữ liệu.
   - Kiểm tra và căn chỉnh Homography sân bóng bằng cách chiếu ngược các đường kẻ sân lên khung hình camera.

---

## 🏗️ Kiến trúc hệ thống (Architecture Pipeline)

```
                       [ Input Broadcast Video ]
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
  [ YOLOv8 Detection ]                          [ Pitch Keypoints Detection ]
  (Ball, GK, Players, Ref)                      (Roboflow 32 Pitch Keypoints)
         │                                                 │
  [ ByteTrack Tracker ]                         [ Perspective Homography ]
  (Assign Persistent IDs)                       (Smoothing + Stickiness Filter)
         │                                                 │
  [ Player Crops Stride ]                                  │
         │                                                 │
  [ SigLIP Vision Embeddings ]                             │
         │                                                 │
  [ UMAP 3D + KMeans Clustering ]                          │
  (Identify Team 0 & Team 1)                               │
         │                                                 │
  [ Goalkeeper Team Assignment ]                           │
  (Spatial Proximity to Team Centroid)                     │
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  │
                 [ 2D Coordinates Transformation ]
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
[ Match Stats Engine ]   [ Radar Mini-map ]       [ Live Video HUD Overlay ]
- Possession % & Streak  - 2D Top-Down Pitch      - Possession Bar
- Kalman Speed & Sprints - Voronoi Control Tanh   - Player Speed Badges
- Distance Covered       - Player & Ball Dots     - Sprint Highlights
- Team Heatmaps                   │                        │
         │                        └───────────┬────────────┘
         │                                    │
 [ Interactive Reports ]              [ Output Video ]
 (.html, .json, .csv)             (Annotated Tactical Video)
```

---

## 📁 Cấu trúc thư mục mã nguồn

```text
football-analysis-system/
├── src/
│   ├── __init__.py               # Khởi tạo package và export các class chính
│   ├── config.py                 # Dataclasses cấu hình: ModelConfig, VisualConfig, PipelineConfig
│   ├── pitch_config.py           # Định nghĩa kích thước chuẩn sân bóng (105x68m) & 32 keypoints
│   ├── detection.py              # Wrapper YOLOv8 detector và ByteTrack tracker
│   ├── team_classifier.py        # Trích xuất SigLIP embedding, UMAP/KMeans clustering & phân đội thủ môn
│   ├── pitch_detector.py         # Nhận diện keypoints sân bóng từ Roboflow Inference API
│   ├── projection.py             # ViewTransformer: Tính ma trận Homography, làm mịn & lọc nhiễu
│   ├── pitch_annotator.py        # Vẽ sân 2D, radar dots, Voronoi diagram (tanh) & chèn mini-map
│   ├── visualizer.py             # Vẽ bounding ellipses, nhãn tracker ID, con trỏ tam giác quả bóng
│   ├── match_stats.py            # Engine thống kê: Possession, Speed/Sprint Kalman, Distance, Heatmaps
│   ├── live_overlay.py           # Vẽ HUD realtime lên video: Possession bar, Speed tags, Sprint alert
│   └── pipeline.py               # Pipeline liên kết toàn bộ luồng xử lý video end-to-end
├── tools/
│   ├── __init__.py
│   ├── cluster_visualizer_3d.py  # Xuất biểu đồ 3D tương tác HTML xem ảnh crop khi click chuột
│   └── test_pitch_homography.py  # Công cụ kiểm tra căn chỉnh homography và keypoints sân bóng
├── football/
│   ├── .gitkeep                  # Thư mục chứa video đầu vào và trọng số mô hình
│   └── ducanh.jpg                # Ảnh mẫu thử nghiệm
├── reports/                      # Thư mục lưu kết quả báo cáo mẫu (HTML, JSON, CSV, video demo)
│   ├── sample_output.mp4         # Video mẫu phân tích hoàn chỉnh sẵn sàng phát ngay khi vào Dashboard
│   ├── sample_match_stats.json   # Dữ liệu JSON chỉ số trận đấu mẫu
│   ├── sample_match_stats_players.csv # Bảng chỉ số chi tiết từng cầu thủ mẫu
│   ├── sample_match_stats.html   # Báo cáo đồ hoạ tương tác HTML mẫu
│   ├── pitch_homography_test.jpg # Ảnh mẫu kiểm tra căn chỉnh sân
│   ├── test_preview.jpg          # Ảnh mẫu kết quả phân tích 1 frame
│   └── player_clusters_3d.html   # Báo cáo 3D phân cụm màu áo tương tác
├── app_dashboard.py              # Web Dashboard phân tích bóng đá (Streamlit)
├── run_dashboard.sh              # Script khởi động nhanh Dashboard trên Linux / macOS
├── run_dashboard.bat             # Script khởi động nhanh Dashboard trên Windows
├── run_dashboard_autostop.sh     # Script quản lý tiến trình Dashboard tự ngắt khi nhàn rỗi
├── main.py                       # Điểm vào dòng lệnh (CLI) chạy phân tích video
├── requirements.txt              # Danh sách các thư viện phụ thuộc
├── .gitignore                    # Bộ lọc file không đưa lên repository (weights, video, venv)
└── README.md                     # Tài liệu hướng dẫn sử dụng chi tiết
```

---

## 🔄 Bảng đối chiếu Notebook sang Module Code

| Cell Notebook gốc | Nội dung / Chức năng | File Module tương ứng |
| :--- | :--- | :--- |
| **Cell 04 - 07** | Tải YOLO, phát hiện cầu thủ / bóng, vẽ Ellipse / Triangle | `src/detection.py`, `src/visualizer.py` |
| **Cell 08** | Khởi tạo ByteTrack tracking các thực thể | `src/detection.py` (`PlayerTracker`) |
| **Cell 10 - 12** | Thu thập crops cầu thủ theo frame stride | `src/team_classifier.py` (`collect_player_crops`) |
| **Cell 13 - 16** | Trích xuất đặc trưng màu áo qua SigLIP Vision Model | `src/team_classifier.py` (`TeamClassifier.extract_features`) |
| **Cell 17 - 19** | Giảm chiều UMAP (3D) và phân cụm KMeans 2 đội | `src/team_classifier.py` (`TeamClassifier.fit/predict`) |
| **Cell 20** | Biểu đồ 3D Plotly tương tác nhúng ảnh Base64 | `tools/cluster_visualizer_3d.py` |
| **Cell 21 - 22** | Gán đội cho thủ môn dựa vào toạ độ không gian | `src/team_classifier.py` (`resolve_goalkeepers_team_id`) |
| **Cell 24 - 29** | Roboflow Inference phát hiện 32 keypoints sân | `src/pitch_detector.py` (`PitchKeypointDetector`) |
| **Cell 30 - 31** | Cấu hình kích thước sân chuẩn & tính toán Homography | `src/pitch_config.py`, `src/projection.py` |
| **Cell 33** | Hàm vẽ biểu đồ Voronoi làm mịn chuyển sắc (`tanh`) | `src/pitch_annotator.py` (`draw_pitch_voronoi`) |
| **Cell 34** | Xử lý từng frame (`process_frame`) vẽ radar mini-map | `src/pipeline.py` |
| **Cell 35 - 36** | Ghi video kết quả ra file | `src/pipeline.py`, `main.py` |
| **Mở rộng mới** | Engine đo tốc độ, bứt tốc, quãng đường, kiểm soát bóng | `src/match_stats.py` |
| **Mở rộng mới** | Live HUD overlay trên video thời gian thực | `src/live_overlay.py` |
| **Mở rộng mới** | Giao diện tương tác trực quan Streamlit | `app_dashboard.py` |

---

## 🚀 Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
- Hệ điều hành: Linux (Ubuntu khuyến nghị), macOS, Windows.
- Python: 3.10 trở lên.
- GPU: Khuyến nghị NVIDIA GPU (hỗ trợ CUDA) để đạt tốc độ xử lý khung hình cao nhất. Hệ thống vẫn hỗ trợ CPU với chế độ fallback tự động.

### 2. Cài đặt môi trường ảo & Thư viện
```bash
# Tạo môi trường ảo (khuyến nghị)
python3 -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate

# Cài đặt các thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

*(Tùy chọn) Cài thêm Roboflow Sports:*
```bash
pip install git+https://github.com/roboflow/sports.git
```
*(Lưu ý: Bộ mã nguồn đã được lập trình sẵn cơ chế fallback tự động. Ngay cả khi chưa cài gói `sports`, mã nguồn vẫn chạy bình thường với các module nội bộ có sẵn).*

### 3. Cấu hình Model Weights & Tự động tải (Auto-Download)
Hệ thống được thiết kế **hoàn toàn tự động và tự phục hồi (self-healing)**:
- **Weights AI (`best.pt`)**: Nếu máy bạn chưa có file trọng số, hệ thống sẽ tự động tải file fine-tuned chuẩn từ [GitHub Release v1.0.0](https://github.com/ducanhdhtb06-hub/football-match-analysis/releases/tag/v1.0.0) về thư mục `football/runs/detect/train/weights/best.pt`. Bạn cũng có thể chỉ định trọng số riêng qua cờ `--weights <đường_dẫn>`.
- **Video mẫu (`test.mp4`)**: Tự động tải từ GitHub Release nếu thư mục `football/` chưa có video đầu vào.
- **Video kết quả demo (`sample_output.mp4`)**: Đã có sẵn trong repo và tự động tải dự phòng nếu bị thiếu.

---

## 💻 Hướng dẫn sử dụng

### ⚡ Khởi chạy nhanh khi Clone về máy mới
```bash
git clone https://github.com/ducanhdhtb06-hub/football-match-analysis.git
cd football-match-analysis

# 1. Khởi tạo môi trường ảo
python3 -m venv .venv
source .venv/bin/activate    # Trên Windows: .venv\Scripts\activate

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. Khởi động Web Dashboard
./run_dashboard.sh           # Trên Windows: run_dashboard.bat
```

---

### 🖥️ 1. Khởi chạy Giao diện Web Dashboard (Streamlit)
Trải nghiệm trực quan toàn bộ tính năng qua giao diện web:
```bash
# Cách 1 (Khuyên dùng - Linux / macOS):
./run_dashboard.sh

# Cách 2 (Windows):
run_dashboard.bat   # Hoặc nhấp đúp chuột vào file run_dashboard.bat

# Cách 3: Chạy trực tiếp qua Streamlit hoặc Python
streamlit run app_dashboard.py
# hoặc
python app_dashboard.py
```
Sau khi khởi chạy, mở trình duyệt web tại: **`http://localhost:8501`**.

> **💡 Điểm nổi bật khi vào Dashboard:**
> - **Hiển thị tức thì:** Kết quả phân tích mẫu cùng 4 thẻ chỉ số và 2 biểu đồ phân tích tương tác hiển thị ngay lập tức trên màn hình.
> - **Vị trí lưu video rõ ràng:** Hiển thị chính xác đường dẫn file video đã lưu cục bộ (trong `reports/dash_runs/...` hoặc `reports/sample_output.mp4`) để mở trực tiếp trên máy.
> - **Tải lên video mới:** Chỉ cần kéo thả file video trận đấu mới (`.mp4`, `.webm`) ở thanh bên trái và bấm **"Bắt đầu phân tích"**.

---

### 🌟 2. Chạy toàn diện dòng lệnh với 1 lệnh duy nhất (`--all`)
Chạy kiểm tra sân bóng, xuất biểu đồ 3D phân cụm màu áo, ảnh kiểm tra preview, và video kết quả hoàn chỉnh có biểu đồ Voronoi:
```bash
python main.py --all --video football/test.mp4 --output football/output.mp4
```

---

### 🎥 3. Các chế độ dòng lệnh (CLI Modes)

#### A. Phân tích video cơ bản (Tracking + Radar Mini-map)
```bash
python main.py --video football/test.mp4 --output football/output.mp4
```

#### B. Bật phân vùng kiểm soát Voronoi trên Mini-map
```bash
python main.py --video football/test.mp4 --output football/output_voronoi.mp4 --voronoi
```

#### C. Thống kê toàn diện trận đấu (`--stats`)
Thu thập dữ liệu tốc độ, bứt tốc, quãng đường, kiểm soát bóng và xuất báo cáo HTML/CSV/JSON:
```bash
python main.py --video football/test.mp4 --output football/output.mp4 --stats --stats-dir reports
```

#### D. Hiển thị HUD chỉ số trực tiếp lên video (`--overlay`)
Vẽ thanh % cầm bóng ở đỉnh video và nhãn tốc độ/bứt tốc trên đầu cầu thủ:
```bash
python main.py --video football/test.mp4 --output football/output_overlay.mp4 --overlay
```

#### E. Kết hợp tối ưu: Overlay + Thống kê + Tăng tốc xử lý
Tăng tốc phân tích bằng cách chỉ suy luận keypoints mỗi 3 khung hình (`--kp-every 3`) và xử lý cách khung (`--stride 2`):
```bash
python main.py --video football/test.mp4 --output football/output_final.mp4 \
               --voronoi --stats --overlay --kp-every 3 --stride 2
```

#### F. Chạy thử nghiệm nhanh (Quick Test)
Xử lý 60 khung hình đầu tiên để kiểm tra kết quả tức thì:
```bash
python main.py --video football/test.mp4 --output football/output_quick.mp4 --max-frames 60 --stride 2
```

Xuất 1 khung hình duy nhất thành ảnh tĩnh:
```bash
python main.py --video football/test.mp4 --test-frame reports/test_preview.jpg
```

---

### 🔬 4. Công cụ chẩn đoán & Trực quan hoá

#### A. Trực quan hoá phân cụm màu áo 3D tương tác (Plotly 3D HTML)
```bash
python main.py --visualize-clusters --video football/test.mp4
```
Mở file `reports/player_clusters_3d.html` trên trình duyệt: rê chuột hoặc bấm vào điểm dữ liệu để xem ảnh crop thực tế của từng cầu thủ.

#### B. Kiểm tra căn chỉnh Homography sân bóng
```bash
python main.py --test-homography --video football/test.mp4
```
File ảnh kiểm tra được lưu tại `reports/pitch_homography_test.jpg`, vẽ lại các đường sân từ không gian 2D ngược về hình ảnh camera để đánh giá độ chuẩn xác của ma trận biến đổi phối cảnh.

---

## 🛠️ Sử dụng như một Python Module

Bạn có thể tích hợp pipeline vào dự án Python của riêng mình:

```python
from src.pipeline import FootballAnalysisPipeline
from src.config import ModelConfig, VisualConfig, PipelineConfig

# 1. Cấu hình mô hình và tham số
model_cfg = ModelConfig(yolo_conf=0.35)
pipeline_cfg = PipelineConfig(
    enable_voronoi=True,
    smooth_homography=True,
    pitch_keypoint_every=3
)

# 2. Khởi tạo pipeline
pipeline = FootballAnalysisPipeline(
    model_config=model_cfg,
    pipeline_config=pipeline_cfg
)

# 3. Fit mô hình phân cụm màu áo 2 đội từ video
pipeline.fit_teams_from_video("football/test.mp4")

# 4. Xử lý video và xuất kết quả
pipeline.process_video(
    source_path="football/test.mp4",
    target_path="football/output_custom.mp4",
    max_frames=300
)
```

---

## 📈 Kết quả báo cáo thống kê (Match Analytics Output)

Khi sử dụng cờ `--stats`, thư mục `reports/` sẽ xuất các tệp:
- `match_stats.html`: Báo cáo đồ họa trực quan bao gồm:
  - Timeline kiểm soát bóng theo từng giây.
  - Biểu đồ tròn tỉ lệ kiểm soát bóng của 2 đội và kiểm soát theo 1/3 sân.
  - Bảng số liệu chi tiết từng cầu thủ: số lần chạm bóng, thời gian giữ bóng, tốc độ cao nhất (km/h), số lần bứt tốc ($\ge 25\text{ km/h}$), quãng đường chạy (m).
  - Bản đồ nhiệt hoạt động (Heatmap) và vị trí trung bình trên sân 2D.
- `match_stats.json`: Dữ liệu thô đầy đủ có cấu trúc để tích hợp vào các hệ thống khác.
- `match_stats_players.csv`: Bảng số liệu dạng bảng theo từng cầu thủ.
- `match_stats_teams.csv`: Bảng tổng kết số liệu của 2 đội bóng.

---

## 📜 Giấy phép (License)
Dự án được phát hành theo giấy phép [MIT License](LICENSE).
