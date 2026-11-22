# Rà soát code và đường dẫn nội bộ V3

- Phạm vi: code đang dùng trong V3/src và bốn DOCX trong V3/outputs. Không đại diện cho mọi bản lưu trữ/V2.
- Không phát hiện AUC gán sẵn thay thế kết quả trong luồng huấn luyện/đánh giá chính. Stage B/I thực sự fit và predict; Stage C tính chỉ số từ OOF predictions.
- Tính lại mean-seed AUC và patient-mean AUC của cả 20 nhãn mô hình từ 12.900 dòng dự đoán: khớp CSV kết quả trong sai số 1e-12.
- Đối chiếu 645 dự đoán GBM với tệp 50 outer refits độc lập đã có: khớp trong sai số 1e-12. Lần rà soát này không chạy mới 50 refits.
- Có số viết sẵn trong văn bản báo cáo ở stage_u_final_review.py và README/assertion ở stage_x_finalize_checked.py. Đây là số mô tả/mốc kiểm tra, không được dùng tạo dự đoán hoặc thay thế phép tính AUC; khi đổi kết quả về sau cần cập nhật hoặc chuyển chúng sang đọc CSV.
- Stage B và G có dùng lại checkpoint sau kiểm tra protocol/hash. Chạy lại toàn bộ script không đồng nghĩa huấn luyện lại từ đầu. Các stage đánh giá, hình, manuscript và kiểm tra đọc outputs có sẵn là đúng chức năng.
- Đã quét mọi XML/relationship trong bốn DOCX, gồm nội dung, thuộc tính tài liệu, header/footer và đích liên kết: xem audit.json để biết chi tiết. Không thay đổi manuscript hoặc số liệu trong lần kiểm tra này.

Kết quả quét đường dẫn: KHÔNG PHÁT HIỆN.

Hai bản Checked khớp SHA256 với FINAL_GATE: False.

Lưu ý: manuscript Checked hiện có SHA256 khác bản ghi FINAL_GATE trước đó; response Checked vẫn khớp. Vì vậy không dùng gate cũ để khẳng định bản manuscript hiện tại đã được kiểm tra bố cục đầy đủ. Quét đường dẫn và tính lại AUC ở báo cáo này là kiểm tra mới, độc lập với gate cũ.
