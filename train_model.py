from sklearn.linear_model import LogisticRegression
import joblib

# Data training sederhana
X = [[50], [60], [70], [80], [90]]
y = [0, 0, 1, 1, 1]  # 0 = Tidak Lulus, 1 = Lulus

# Latih model
model = LogisticRegression()
model.fit(X, y)

# Simpan model ke file
joblib.dump(model, 'model.pkl')
