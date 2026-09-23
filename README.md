# Đóng khung phụ đề bằng PaddleOCR mobile

Detection-only với **PP-OCRv6_small_det** (PaddleOCR 3.7.0). Không đọc chữ, không xuất SRT; bộ lọc hình học/vị trí không xác minh được ngôn ngữ. Model được chỉ định rõ, không dùng model server mặc định.

## Cài đặt (Python 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
# Chọn đúng MỘT bản PaddlePaddle:
# NVIDIA GPU, driver hỗ trợ CUDA 12.6:
.\.venv\Scripts\python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# Hoặc CPU:
# .\.venv\Scripts\python -m pip install paddlepaddle==3.3.0
```

Không cần cài FFmpeg hệ thống: `imageio-ffmpeg` cung cấp binary. Lần chạy đầu cần Internet để tải model, lưu ở `.cache/`. Có thể dùng `--model-dir` trỏ đến thư mục model đã tải. Không cài đồng thời `paddlepaddle` và `paddlepaddle-gpu` trong một môi trường.

## Chạy

```powershell
.\.venv\Scripts\python detect_subtitles.py --input "AI Engineer test.mp4" --output outputs/result.mp4 --device gpu:0
# Thử đoạn có phụ đề:
.\.venv\Scripts\python detect_subtitles.py --input "AI Engineer test.mp4" --output outputs/preview.mp4 --device gpu:0 --start 35 --duration 5
# CPU: thay --device gpu:0 bằng --device cpu
```

Output cùng tên:

- `.mp4`: H.264, giữ timestamp nguồn và kích thước; audio đầu tiên được trim theo PTS và encode AAC 192 kbps. Video không audio vẫn xử lý được.
- `.jsonl`: mỗi frame một dòng, `frame_index` gốc (0-based), `source_timestamp_seconds` tính từ frame đầu của video nguồn, `output_timestamp_seconds` tính từ frame đầu được chọn, `boxes_xyxy` là tọa độ box đã vẽ trong ảnh gốc.
- `.metrics.json`: cấu hình máy, phiên bản thư viện, số frame, số đoạn phụ đề được giữ (`subtitle_segments`), số đoạn ngắn bị loại (`discarded_short_segments`), thông số video và thời gian từng bước.

Không ghi đè nếu thiếu `--overwrite`; không cho phép dùng đường dẫn input làm output. File tạm tự dọn khi gặp lỗi. GPU không khả dụng sẽ báo lỗi, không âm thầm chuyển CPU.

## Cách xử lý

Quét mọi frame, crop 45% đáy trước detector. Model resize tối đa cạnh dài 960 px, không phóng lớn ROI nhỏ. `--roi-bottom 0.25` đến `0.45` điều chỉnh vùng quét; ROI hẹp có thể bỏ phụ đề nằm cao trong video dọc.

Polygon được quy đổi về ảnh gốc, lọc confidence/kích thước/góc, rồi ghép mảnh chữ cùng dòng. Pipeline chạy hai lượt: lượt đầu detect và gom các frame liên tiếp có cùng dòng phụ đề bằng IoU và dấu vết cạnh chữ trên một vùng ảnh cố định; lượt sau đọc lại video gốc để vẽ. Đoạn chỉ được giữ khi có ít nhất 3 frame được model detect thật và kéo dài ít nhất 0,8 giây tính theo timestamp; frame được bù không tăng số detection. Đoạn chạm mép khoảng bị cắt bởi `--start` hoặc `--duration` được giữ vì có thể chỉ là một phần phụ đề. Mỗi đoạn được giữ có một box cố định từ phân vị 10%/90% của các detection, nên box đứng yên trong suốt đoạn. Có thể bù tối đa 2 frame mất detection khi hình dạng chữ vẫn giống và detection xuất hiện trở lại. Khi câu thay đổi hoặc chữ biến mất, đoạn được kết thúc. Box xanh có padding 3 px, nét 2 px.

GPU dùng batch 8, CPU batch 1; đổi bằng `--batch-size`. Nếu thiếu VRAM, giảm batch xuống 4/2/1. Bộ nhớ ảnh giới hạn theo batch; tọa độ frame và box của các đoạn được giữ đến lượt vẽ thứ hai. Video nguồn được giải mã hai lần. Dùng `--min-subtitle-seconds` và `--min-detection-frames` để đổi hai ngưỡng lọc, đặt từng ngưỡng thành `0` để tắt kiểm tra tương ứng. `--box-threshold`, `--pixel-threshold`, `--det-size`, `--padding`, `--cpu-threads` đều chỉnh được; xem `--help`.

CPU mặc định tắt oneDNN: kiểm thử PP-OCRv6_small_det với PaddlePaddle 3.3.0 Windows gặp lỗi `ConvertPirAttribute2RuntimeAttribute` khi bật. Chỉ thử `--enable-mkldnn` trên runtime đã xác nhận hỗ trợ; nếu lỗi, bỏ cờ này. Benchmark phải ghi rõ trạng thái oneDNN để so sánh công bằng.

Chữ trên áo/biển hiệu trong ROI vẫn có thể bị nhận nhầm. Phụ đề ngoài ROI sẽ bị bỏ qua. Pipeline không nhận diện ngữ nghĩa, không bảo đảm phân biệt mọi chữ cảnh nền với phụ đề. File có rotation metadata cần được chuẩn hóa trước nếu pixel lưu trữ không đúng chiều hiển thị.

## Benchmark

```powershell
.\.venv\Scripts\python benchmark.py --input "AI Engineer test.mp4" --devices cpu gpu:0 --duration 30 --repeats 3 --output-dir outputs/benchmark
```

Mỗi lần chạy dùng process riêng, warm-up 3 batch. Lưu video/log/metrics từng lần; `summary.csv`, `summary.json`, `report.md` tổng hợp median/min/max. Dùng thư mục mới cho mỗi đợt benchmark; `--report-only` tái tạo báo cáo từ kết quả cũ.

`processing_seconds` là wall time xử lý gồm hai lượt decode, detect, gom đoạn, encode và ghép audio, không gồm khởi tạo model, warm-up, bước decode xác minh output. `inference_seconds` bao gồm tiền/hậu xử lý của PaddleX và được đồng bộ GPU. Latency p95/median tính **theo batch**, không phải độ trễ từng frame. `amortized_inference_ms_per_frame` là tổng thời gian inference chia số frame. `total_seconds_including_init_warmup_verify` phản ánh tổng thời gian kể cả chuẩn bị/kiểm tra. Seek đến `--start` hiện decode và bỏ các frame trước đó; hãy benchmark cùng start để so sánh công bằng.

Các máy không đo được được ghi **chưa đủ dữ liệu**, không suy tốc độ từ TFLOPS/VRAM. Để thêm ước lượng có căn cứ, truyền `--estimates estimates.json`, dạng mảng:

```json
[
  {
    "hardware": "Tesla T4",
    "model": "PP-OCRv6_small_det",
    "backend": "ghi backend thực tế",
    "source": "URL hoặc đường dẫn benchmark có thể kiểm chứng",
    "assumptions": "ghi ROI, batch, kích thước inference và cách suy ra khoảng thời gian",
    "seconds_low": 1,
    "seconds_high": 2,
    "video_duration_seconds": 124.433333
  }
]
```

Các số trong ví dụ chỉ minh họa schema, **không phải ước lượng hiệu năng**. Không dùng kết quả CPU trên Apple Silicon làm số liệu GPU Metal/ANE.

## Kiểm thử

```powershell
.\.venv\Scripts\python -m pip install pytest
.\.venv\Scripts\python -m pytest -q
```

Unit test bao phủ ROI, ghép dòng, mất detection ngắn, đổi câu, mất chữ và chuyển cảnh. Integration test dùng detector giả nhưng codec thật để kiểm tra CFR/VFR, audio, trim và bảo vệ output; không thay thế kiểm tra chất lượng model trên video thực tế.

Nguồn: [PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md), [TextDetection API](https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html), [PaddlePaddle Windows](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip.html).
