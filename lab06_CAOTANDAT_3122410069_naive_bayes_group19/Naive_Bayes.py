import numpy as np

class Naive_Bayes:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
        # Tìm các lớp (classes) duy nhất
        self.classes = np.unique(y_train)
        
        # Khởi tạo các tham số Mean, Variance, và Prior cho từng lớp
        self.params = {}

        # Huấn luyện (Tính toán thống kê) ngay khi khởi tạo
        self.fit()

    def fit(self):
        for c in self.classes:
            # Lọc lấy các dòng thuộc class c
            X_c = self.X_train[self.y_train == c]
            
            # Tính mean, var, prior
            mean = X_c.mean(axis=0)
            var = X_c.var(axis=0) + 1e-6 # Cộng thêm epsilon để tránh var = 0
            prior = X_c.shape[0] / self.X_train.shape[0]
            
            # Lưu lại tham số
            self.params[c] = {
                "mean": mean,
                "var": var,
                "log_prior": np.log(prior) # Lưu luôn dạng Log
            }

    def calculate_log_likelihood(self, X, mean, var):
        # Công thức tính log likelihood của Gaussian:
        # log(P(x|c)) = -0.5 * log(2 * pi * var) - 0.5 * ((x - mean)^2 / var)
        
        epsilon = 1e-6 # Tránh log(0) nếu cần, dù var đã xử lý rồi
        
        term1 = -0.5 * np.log(2 * np.pi * var)
        term2 = -0.5 * ((X - mean) ** 2 / var)
        
        return term1 + term2

    def predict(self, X):
        # X ở đây là toàn bộ ma trận Test data (N_samples, N_features)
        # Chúng ta sẽ dự đoán cho tất cả các dòng cùng lúc
        
        y_pred = []
        
        # Tạo một list chứa xác suất của từng class cho mọi điểm dữ liệu
        posteriors = []

        for c in self.classes:
            mean = self.params[c]["mean"]
            var = self.params[c]["var"]
            log_prior = self.params[c]["log_prior"]
            
            # Tính Log Likelihood cho toàn bộ ma trận X với class c
            # log_likelihood shape: (N_samples, N_features)
            log_likelihood = self.calculate_log_likelihood(X, mean, var)
            
            # Tổng hợp log-likelihood theo từng tính năng (axis=1) để ra xác suất của dòng
            # Posterior = Log Prior + Sum(Log Likelihoods)
            posterior = log_prior + np.sum(log_likelihood, axis=1)
            
            posteriors.append(posterior)
            
        # posteriors đang là list các array. Chuyển về ma trận (N_classes, N_samples)
        posteriors = np.array(posteriors)
        
        # Tìm class có xác suất cao nhất cho mỗi mẫu (argmax theo trục 0 - trục các class)
        # Kết quả trả về là index của class trong self.classes
        best_class_indices = np.argmax(posteriors, axis=0)
        
        # Map từ index về tên class gốc
        return self.classes[best_class_indices]

    def score(self, X_test, y_test):
        # Hàm này thay thế cho hàm test cũ, tính accuracy
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy, predictions