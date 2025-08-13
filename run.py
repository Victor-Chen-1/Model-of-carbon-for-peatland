from camp_model import default_params, CaMPModel
import numpy as np

# 1) 参数
params = default_params()
model = CaMPModel(params)

# 2) 气候序列（示例：200 年）
years = 200
mat_series = (np.sin(np.linspace(0, 8*np.pi, years))*1.5 + 0.5).tolist()  # 年均温 °C
dc_series  = (np.clip(300 + 50*np.sin(np.linspace(0, 4*np.pi, years)) + np.linspace(0, 60, years), 0, 800)).tolist()

# 3) 初始化（使用前 30 年平均作为“气候态”）
init_mat = float(np.mean(mat_series[:30]))
init_dc  = float(np.mean(dc_series[:30]))
model.initialize_from_steady_state(init_mat, init_dc)

# 4) 前向模拟
df = model.simulate(years, mat_series, dc_series, seed=42)
df.to_csv("my_run.csv", index=False)
print(df.head())
