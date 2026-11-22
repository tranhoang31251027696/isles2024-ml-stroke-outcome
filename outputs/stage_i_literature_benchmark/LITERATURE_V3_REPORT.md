# Literature benchmark — V3

Tái huấn luyện 8 cấu hình literature: 129 ca, cùng 22 biến, 5 seed × 10 outer folds như GBM V3. 400 lần fit.
Dùng missing gốc v7; binary điền mode, biến khác median, scaler fit trên training fold. Hai XGBoost dùng XGBClassifier thật.

| Hạng theo điểm AUC | Mô hình | Mean seed OOF AUC ± SD | 95% CI của mean seed AUC |
|---:|---|---:|---|
| 1 | Fixed Cand4 GBM (this study) | 0.7599 ± 0.0184 | 0.6729–0.8394 |
| 2 | Bogey et al. - RF (d=5) | 0.7438 ± 0.0054 | 0.6498–0.8295 |
| 3 | Bogey et al. - GBM (d=3) | 0.7434 ± 0.0162 | 0.6522–0.8288 |
| 4 | Otieno et al. - XGBoost (d=6) | 0.7315 ± 0.0172 | 0.6403–0.8181 |
| 5 | Liu et al. - XGBoost (d=4) | 0.7298 ± 0.0110 | 0.6360–0.8189 |
| 6 | Li et al. - ElasticNet LR | 0.7144 ± 0.0128 | 0.6130–0.8111 |
| 7 | Otieno et al. - SVM RBF | 0.7110 ± 0.0114 | 0.6097–0.8104 |
| 8 | Otieno et al. - mLR (L2) | 0.7080 ± 0.0127 | 0.6049–0.8088 |
| 9 | Otieno et al. - ANN (100-50) | 0.6278 ± 0.0111 | 0.5278–0.7263 |

Cấu hình literature được giữ cố định, GBM chính cố định Cand4 sau khảo sát trên cohort; bảy họ khác dùng inner CV. Đây là benchmark bổ sung, không chứng minh superiority hay equal tuning.
Xếp hạng là thứ tự điểm ước lượng, không tự chứng minh khác biệt thống kê. CI/p dùng paired patient bootstrap có điều kiện trên dự đoán đã fit, 2.000 lần; p được điều chỉnh Holm cho 8 so sánh với GBM.
Đây là các cấu hình tái triển khai trên cohort hiện tại, không phải hiệu năng đã công bố của các bài nguồn và không phải external validation.
Manuscript/response chưa được sửa trong lần chạy này.
