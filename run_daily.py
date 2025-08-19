
# run_daily.py
# -*- coding: utf-8 -*-

"""
使用说明（简要）
----------------
1) 准备数据：
   - DWD 气象日数据：例如 `produkt_klima_tag_19850301_20241231_03086`（以 `;` 分隔）
     必要字段：`MESS_DATUM`（YYYYMMDD）、`TMK`（日平均温度 °C）、`RSK`（日降水量 mm）。
   - 可选的场地参数 CSV（列名示例）：NEE,a,b,FCH4Max,F_10_p,F_10_n,k_base,Q_10,IniCStk,dlt_WT。

2) 修改下方 CONFIG 区域中的文件路径和参数（尤其是 `CLIMATE_FILE`、`SITE_PARAM_FILE`）。

3) 运行：
   python run_daily.py

输出：
   - daily_results.csv：逐日的池值、通量、DC、WT 等。

注意：
   - 本脚本内置的 DC 计算实现遵循 CFFDRS 的公式（Van Wagner 1985/1987），
     与 R 包 cffdrs 的 dcCalc 等价（见注释与文献）。
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Sequence
from camp_model import default_params, CaMPModel, applied_decay_rate, water_table_from_dc, ch4_flux_gCm2_day

# -------------------------------
# CONFIG
# -------------------------------
CLIMATE_FILE = "produkt_klima_tag_19850301_20241231_03086.txt"  # 修改为你的实际路径或文件名
SITE_PARAM_FILE = "site_params.csv"  # 例如: "site_params.csv"；若无则留空
STATION_LAT = 52.0      # 用于 DC 日照因子（约等于柏林纬度，可按站点修正）
START_DC = 15.0         # DC 起始值（若无越冬处理）
SEED = 42

# 当提供 k_base 时，对默认分解速率进行缩放的参考池（默认使用 acrotelm_fast 的默认值 0.15）
KBASE_REFERENCE_POOL = "acrotelm_fast"
DEFAULT_KREF_FOR_REFERENCE = 0.15  # 与 camp_model.default_params() 中一致

# -------------------------------
# DC (Drought Code) 计算
#   依据 Van Wagner (1985/1987)，实现与 R 包 cffdrs::dcCalc 等价。
#   公式参考：
#     - 有效降水：rw = 0.83*P - 1.27 （若 P <= 2.8 则不进入湿润过程）
#     - 潜在蒸散：pe = (0.36*(T+2.8) + fL[mon]) / 2 （按纬度范围使用不同 fL）
#     - SMI = 800*exp(-DC/400)
#     - 湿润后 DC 值：dr0 = DC_prev - 400*ln(1 + 3.937*rw/SMI)（再截断到 >=0）
#     - 最终 DC：dc = max(dr, 0) + max(pe, 0)
# -------------------------------
def compute_dc_series(
    dates: Sequence[pd.Timestamp],
    temp_C: Sequence[float],
    precip_mm: Sequence[float],
    lat_deg: float = 52.0,
    start_dc: float = 15.0,
    lat_adjust: bool = True,
) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(dates)).dt.tz_localize(None)
    T = pd.Series(temp_C).astype(float).copy()
    P = pd.Series(precip_mm).astype(float).copy()

    # 处理缺测：温度线性插值，降水缺测置 0
    T.replace([-999, -9999], np.nan, inplace=True)
    P.replace([-999, -9999], np.nan, inplace=True)
    T = T.interpolate(limit_direction="both")
    P = P.fillna(0.0)

    # 月份索引（1-12）
    mon = dates.dt.month.values

    # 日照（月）因子（Van Wagner & Pickett 1985；cffsdrs 源码 fl01/fl02）
    fl01 = np.array([-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6])  # 北纬 >20°
    fl02 = np.array([ 6.4,  5.0,  2.4, 0.4, -1.6, -1.6,-1.6,-1.6,-1.6, 0.9,  3.8,  5.8])  # 南纬 < -20°

    # 逐日潜在蒸散 pe
    T_c = np.maximum(T, -2.8)  # 低温截断
    base = 0.36 * (T_c + 2.8)

    if lat_adjust:
        # 北纬 >20°
        if lat_deg > 20:
            pe = (base + fl01[mon - 1]) / 2.0
        # 南纬 <-20°
        elif lat_deg < -20:
            pe = (base + fl02[mon - 1]) / 2.0
        # 赤道附近 |lat|<=20°
        else:
            pe = (base + 1.4) / 2.0
    else:
        pe = (base + fl01[mon - 1]) / 2.0  # 默认按北纬表
    pe = np.maximum(pe, 0.0)

    # 逐日迭代
    dc = np.zeros(len(T), dtype=float)
    prev = float(start_dc)
    for i in range(len(T)):
        ra = float(P.iloc[i])
        # 若降水不足阈值，则不进入湿润过程
        if ra <= 2.8:
            dr = prev
        else:
            rw = 0.83 * ra - 1.27            # 有效降水
            smi = 800.0 * math.exp(-prev/400.0)
            dr0 = prev - 400.0 * math.log(1.0 + 3.937 * rw / smi)
            dr = max(dr0, 0.0)
        # 干燥项叠加
        cur = max(dr + float(pe[i]), 0.0)
        dc[i] = cur
        prev = cur
    return dc

# -------------------------------
# 数据读取：DWD 日气候文本
# -------------------------------
def load_dwd_daily(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8", engine="python", skipinitialspace=True)
    df.columns = df.columns.str.strip()  # 去掉列名前后的空格
    # 必要列
    needed = ["MESS_DATUM", "TMK", "RSK"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"文件缺少必要列: {missing}")

    df["date"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d", errors="coerce")
    df["TMK"] = pd.to_numeric(df["TMK"], errors="coerce")
    df["RSK"] = pd.to_numeric(df["RSK"], errors="coerce")

    df = df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    # 质量控制（可选）：若存在 QN_* 可在此筛选
    # df = df[df["QN_3"].isin([1,3]) & df["QN_4"].isin([1,3])]

    # 替换无效值，例如 -999
    df["TMK"] = df["TMK"].replace([-999, -9999], np.nan)
    df["RSK"] = df["RSK"].replace([-999, -9999], np.nan)

    # 简单插补
    df["TMK"] = df["TMK"].interpolate(limit_direction="both")
    df["RSK"] = df["RSK"].fillna(0.0)
    return df[["date", "TMK", "RSK"]]

# -------------------------------
# 可选：读取场地参数（覆盖部分模型参数）
# -------------------------------
def load_site_params(path: str) -> Optional[pd.Series]:
    if path is None or not os.path.exists(path):
        return None
    sp = pd.read_csv(path)
    # 兼容可能只有一行的配置
    if len(sp) == 0:
        return None
    row = sp.iloc[0]
    return row

# -------------------------------
# 每日版模型（对 CaMPModel 的轻量派生）
#   - 将年分解率转换为日分解率：k_day ≈ k_year / 365
#   - 火灾概率按天计算：p_fire_day = 1/(return_interval_yr*365)
#   - 甲烷通量保持“日”量纲（不再 *365）
# -------------------------------
class CaMPModelDaily(CaMPModel):
    def simulate_daily(
        self,
        dates: Sequence[pd.Timestamp],
        mat_C_series: Sequence[float],
        dc_series: Sequence[float],
        dlt_WT_series: Optional[Sequence[float]] = None,
        seed: int = 123,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        records = []
        n = len(mat_C_series)
        mat = np.array(mat_C_series, dtype=float)
        dc = np.array(dc_series, dtype=float)

        if dlt_WT_series is None:
            dlt_WT = np.zeros(n, dtype=float)
        else:
            dlt_WT = np.array(dlt_WT_series, dtype=float)

        # 日概率
        p_fire_day = 0.0
        if self.p.fire.return_interval_yr > 0:
            p_fire_day = 1.0 / (self.p.fire.return_interval_yr * 365.0)

        chain = ["litter_foliage", "litter_woody", "dead_roots", "acrotelm_fast", "acrotelm_slow", "catotelm"]

        for t in range(n):
            WT = water_table_from_dc(float(dc[t]), self.p.wt) + float(dlt_WT[t])

            # 1) 落叶/凋落物流转（与年版一致）
            live_to_dead = pd.Series(0.0, index=self.p.pool_names)
            for live, npp in self.p.NPP.items():
                self.pools[live] += npp / 365.0  # NPP 改为日累积
                for dst, frac in self.p.live_turnover.get(live, {}).items():
                    amount = self.pools[live] * frac
                    self.pools[live] -= amount
                    live_to_dead[dst] += amount

            # 2) 分解（将年率换算为“日”率）
            atm_release = 0.0
            to_next_flux = pd.Series(0.0, index=self.p.pool_names)

            for pool in chain:
                dpar = self.p.decay[pool]
                a_k_year = applied_decay_rate(dpar.k_ref, float(mat[t]), dpar.q10)
                a_k_day = a_k_year / 365.0
                D = a_k_day * self.pools[pool]  # 当日分解量
                route = self.p.routing[pool]
                to_atm = D * route.to_atm
                to_next = D * route.to_next
                atm_release += to_atm
                to_next_flux[pool] = to_next
                self.pools[pool] += live_to_dead.get(pool, 0.0)
                self.pools[pool] -= D

            # 3) 级联转移
            for i, pool in enumerate(chain[:-1]):
                nxt = chain[i+1]
                self.pools[nxt] += to_next_flux[pool]

            # 4) 甲烷通量（保持“日”单位）
            F_ch4_day = ch4_flux_gCm2_day(WT, self.p.ch4)

            # 5) 火扰动（按日概率）
            fire_event = False
            if p_fire_day > 0 and (rng.uniform() < p_fire_day):
                fire_event = True
                for live in ["live_trees", "live_shrubs", "live_sedge", "live_moss"]:
                    burned = self.pools[live] * self.p.fire.combust_frac_live
                    self.pools[live] -= burned
                    atm_release += burned
                for lit in ["litter_foliage", "litter_woody"]:
                    burned = self.pools[lit] * self.p.fire.combust_frac_litter
                    self.pools[lit] -= burned
                    atm_release += burned
                depth = self.p.fire.avg_burn_depth_cm
                # 注意：此处的可燃泥炭量估算与年版一致
                from camp_model import C_cum
                peat_C = C_cum(depth, self.p.cdens)
                for pool in ["acrotelm_fast", "acrotelm_slow"]:
                    take = min(self.pools[pool], peat_C * 0.5)
                    self.pools[pool] -= take
                    atm_release += take

            rec = {
                "date": pd.Timestamp(dates[t]).date(),
                "doy": pd.Timestamp(dates[t]).timetuple().tm_yday,
                "year": pd.Timestamp(dates[t]).year,
                "MAT_C": float(mat[t]),
                "DC": float(dc[t]),
                "WT_cm": float(WT),
                "atm_release": float(atm_release),
                "CH4_gCm2_day": float(F_ch4_day),
                "fire": int(fire_event),
            }
            for name in self.p.pool_names:
                rec[f"pool_{name}"] = float(self.pools[name])
            records.append(rec)

        return pd.DataFrame.from_records(records)

# -------------------------------
# 主流程
# -------------------------------
def main():
    # 1) 读取气象
    clima = load_dwd_daily(CLIMATE_FILE)
    # 2) 计算 DC
    dc = compute_dc_series(clima["date"].values, clima["TMK"].values, clima["RSK"].values,
                           lat_deg=STATION_LAT, start_dc=START_DC)
    clima["DC"] = dc

    # 3) 构建模型参数
    params = default_params()

    # 3.1 可选：覆盖 CH4/分解/初始池等参数
    site = load_site_params(SITE_PARAM_FILE)
    if site is not None:
        # CH4 参数
        if "FCH4Max" in site:
            params.ch4.F_CH4max_gCm2_day = float(site["FCH4Max"])
        if "F_10_p" in site:
            params.ch4.F10_up = float(site["F_10_p"])
        if "F_10_n" in site:
            params.ch4.F10_down = float(site["F_10_n"])

        # Q10 覆盖到所有分解池
        if "Q_10" in site and not pd.isna(site["Q_10"]):
            q10v = float(site["Q_10"])
            for pool in ["litter_foliage", "litter_woody", "dead_roots", "acrotelm_fast", "acrotelm_slow", "catotelm"]:
                params.decay[pool].q10.Q10 = q10v

        # k_base：相对缩放所有 k_ref（以参考池默认值为基准）
        if "k_base" in site and not pd.isna(site["k_base"]):
            k_base = float(site["k_base"])
            scale = k_base / DEFAULT_KREF_FOR_REFERENCE
            for pool in ["litter_foliage", "litter_woody", "dead_roots", "acrotelm_fast", "acrotelm_slow", "catotelm"]:
                params.decay[pool].k_ref *= scale

        # 可选：初始大库（例如 catotelm）
        if "IniCStk" in site and not pd.isna(site["IniCStk"]):
            ini_cat = float(site["IniCStk"])
        else:
            ini_cat = None

        # 可选：日尺度 WT 微调项（常数或列）
        if "dlt_WT" in site and not pd.isna(site["dlt_WT"]):
            dlt_wt_const = float(site["dlt_WT"])
        else:
            dlt_wt_const = 0.0
    else:
        ini_cat = None
        dlt_wt_const = 0.0

    # 4) 初始化（用首个完整年的气候均值作为稳态起点）
    if len(clima) >= 365:
        init_slice = slice(0, 365)
    else:
        init_slice = slice(0, len(clima))
    init_mat = float(clima["TMK"].iloc[init_slice].mean())
    init_dc = float(clima["DC"].iloc[init_slice].mean())

    model = CaMPModelDaily(params)
    model.initialize_from_steady_state(init_mat, init_dc)

    if ini_cat is not None:
        # 覆盖 catotelm 初始值（若提供）
        model.pools["catotelm"] = float(ini_cat)

    # 5) 前向逐日模拟
    dates = clima["date"].values
    mat_series = clima["TMK"].values
    dc_series = clima["DC"].values

    dlt_wt_series = np.full(len(clima), dlt_wt_const, dtype=float)

    df = model.simulate_daily(dates, mat_series, dc_series, dlt_wt_series, seed=SEED)

    # 确保 date 列类型一致
    df["date"] = pd.to_datetime(df["date"])
    clima["date"] = pd.to_datetime(clima["date"])

    # 合并（保留原气候列）
    out = df.merge(clima, how="left", on="date")
    out.to_csv("daily_results.csv", index=False)
    print("Saved -> daily_results.csv")
    print(out.head())


if __name__ == "__main__":
    main()
