# Script detect phụ đề bằng PaddleOCR mobile

## Phương án chốt

- Dùng **`PP-OCRv6_small_det`**, dòng mobile/desktop mới nhất theo [tài liệu PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md).
- **Detection-only**, theo lựa chọn của bạn; không chạy recognition hoặc xuất SRT.
- Video đầu vào hiện có: **720 × 1280, 30 FPS, 124,43 giây**. Máy có **RTX 4060 Laptop 8 GB**, Python 3.11; chưa cài PaddleOCR/PaddlePaddle.
- Kết quả chính: video có box xanh ôm từng dòng phụ đề, giữ âm thanh, kèm tọa độ và báo cáo thời gian.

## Các script và cách chạy

- `detect_subtitles.py`: đọc video, detect, ổn định box và xuất MP4. Dùng API `TextDetection` của PaddleOCR 3.x; khóa phiên bản dependencies đã kiểm thử.
- `benchmark.py`: đo CPU/GPU và tổng hợp báo cáo JSON/CSV/Markdown.
- README tiếng Việt hướng dẫn tạo môi trường, cài PaddlePaddle GPU/CPU và chạy. GPU được yêu cầu mà không khả dụng thì báo lỗi rõ ràng.

Lệnh chạy mặc định:

```powershell
python detect_subtitles.py --input "AI Engineer test.mp4" --output "outputs/result.mp4" --device gpu:0
python benchmark.py --input "AI Engineer test.mp4" --devices cpu gpu:0 --output-dir "outputs/benchmark"
```

CLI cho phép đổi model, ROI, batch size, ngưỡng detection và đoạn video cần xử lý; không xây web/API.

## Pipeline xử lý

- Cắt **45% đáy ảnh** trước inference, phù hợp vị trí phụ đề thấy trong video mẫu. Cho phép chỉnh `--roi-bottom` trong khoảng 0,25–0,45.
- Detect mọi frame để không bỏ câu ngắn; batch mặc định 8 trên GPU, 1 trên CPU. Giữ bộ đệm giới hạn, không nạp toàn bộ video vào RAM.
- Đổi polygon về tọa độ ảnh gốc; lọc theo confidence, kích thước và độ nằm ngang; gộp các mảnh cùng dòng, giữ riêng các dòng khác nhau.
- Ghép box giữa các frame theo độ chồng lấp và vị trí. Làm mượt trong cùng đoạn phụ đề; dùng bộ đệm nhìn trước 2 frame để nối lỗi mất detection ngắn khi hình ảnh vùng chữ vẫn tương đồng.
- Khi vùng chữ thay đổi hoặc biến mất, kết thúc đoạn cũ; không kéo box cũ sang câu mới hoặc qua cảnh chuyển.
- Vẽ box xanh, padding 3 px, nét 2 px ở độ phân giải gốc. Không chèn chữ nhận dạng hay bảng debug vào video bàn giao.
- Dùng FFmpeg để xuất H.264, giữ kích thước, thời gian phát và âm thanh; xử lý cả video không có audio. Với video VFR, giữ timestamp nguồn.
- Xuất JSONL gồm frame index, timestamp và danh sách box; ghi riêng cấu hình chạy, model và phiên bản thư viện.

## Đo hiệu năng và kiểm thử

- Đo riêng khởi tạo model, warm-up, inference, hậu xử lý và xuất video; báo tổng thời gian thực tế, FPS xử lý và latency inference. Đồng bộ GPU khi đo.
- Benchmark CPU/GPU trên cùng đoạn 30 giây sau warm-up, chạy 3 lần; đồng thời ghi thời gian chạy toàn bộ video bàn giao.
- Bảng phần cứng theo đề phân biệt **đo thực tế** và **ước lượng**. Ước lượng phải ghi nguồn, backend và giả định; cấu hình thiếu cơ sở được ghi “chưa đủ dữ liệu”, không tự tạo hệ số tốc độ.
- Test logic ghép dòng, quy đổi tọa độ ROI, mất detection 1–2 frame, đổi câu, mất phụ đề và chuyển cảnh.
- Kiểm tra video thực tế ở đầu/giữa/cuối và các điểm đổi câu: box ôm chữ, không nối hai dòng, không lưu box sau khi chữ biến mất.
- Xác minh output đọc được toàn bộ, đúng độ phân giải, không lệch tiếng/hình và thời lượng sai khác không quá một frame.

## Giới hạn đã chọn

Detection-only không xác minh được ngôn ngữ; chữ khác nằm trong vùng phụ đề vẫn có thể bị đóng box. Phạm vi bàn giao là script, video và báo cáo trong máy; không tự gửi email hoặc upload Drive.
