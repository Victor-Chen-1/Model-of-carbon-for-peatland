
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
import pandas as pd
import math

# -------------------------------
# Configuration dataclasses
# -------------------------------

@dataclass
class Q10Params:
    Q10: float = 2.0
    ref_temp_C: float = 10.0

@dataclass
class DecayParams:
    k_ref: float
    q10: Q10Params

@dataclass
class PoolRouting:
    to_atm: float
    to_next: float

@dataclass
class CH4Params:
    F10_up: float = 0.32
    F10_down: float = 2.6
    WT_opt_cm: float = -10.0
    F_CH4max_gCm2_day: float = 0.12

@dataclass
class WTParams:
    b_wc: float

@dataclass
class CarbonDensityParams:
    beta0: float = -3.0
    beta1: float =  0.03
    beta2: float = -0.0002
    gamma: float = 60.0
    delta: float = 0.012

@dataclass
class WildfireParams:
    return_interval_yr: int = 150
    combust_frac_live: float = 0.8
    combust_frac_litter: float = 0.6
    avg_burn_depth_cm: float = 5.0

@dataclass
class ModelParams:
    pool_names: List[str] = field(default_factory=lambda: [
        "live_trees", "live_shrubs", "live_sedge", "live_moss",
        "litter_foliage", "litter_woody", "dead_roots",
        "acrotelm_fast", "acrotelm_slow", "catotelm"
    ])
    decay: Dict[str, DecayParams] = field(default_factory=dict)
    routing: Dict[str, PoolRouting] = field(default_factory=dict)
    live_turnover: Dict[str, Dict[str, float]] = field(default_factory=dict)
    NPP: Dict[str, float] = field(default_factory=dict)
    ch4: CH4Params = field(default_factory=CH4Params)
    wt: WTParams = field(default_factory=lambda: WTParams(b_wc=-15.0))
    cdens: CarbonDensityParams = field(default_factory=CarbonDensityParams)
    fire: WildfireParams = field(default_factory=WildfireParams)
    catotelm_years: int = 8000

# -------------------------------
# Helper functions
# -------------------------------

def temp_modifier_q10(mat_C: float, q10: Q10Params) -> float:
    return math.exp(((mat_C - q10.ref_temp_C) * math.log(q10.Q10)) / 10.0)

def applied_decay_rate(k_ref: float, mat_C: float, q10: Q10Params) -> float:
    return k_ref * temp_modifier_q10(mat_C, q10)

def water_table_from_dc(dc: float, wt_params: WTParams) -> float:
    return 0.045 * dc + wt_params.b_wc

def C_inst(z_cm: float, p: CarbonDensityParams) -> float:
    return math.exp(p.beta0 + p.beta1 * z_cm + p.beta2 * (z_cm ** 2))

def C_cum(z_cm: float, p: CarbonDensityParams) -> float:
    return p.gamma * (1.0 - math.exp(-p.delta * z_cm))

def ch4_flux_gCm2_day(WT_cm: float, ch4: CH4Params) -> float:
    d = ch4.WT_opt_cm - WT_cm
    if d >= 0:
        F10 = ch4.F10_down
    else:
        F10 = ch4.F10_up
    return ch4.F_CH4max_gCm2_day * (F10 ** (abs(d) / 10.0))

# -------------------------------
# Core model
# -------------------------------

class CaMPModel:
    def __init__(self, params: ModelParams):
        self.p = params
        self.pools = pd.Series(0.0, index=self.p.pool_names)
        self.history = []

    def initialize_from_steady_state(self, mat_C: float, dc_clim: float):
        for live in ["live_trees", "live_shrubs", "live_sedge", "live_moss"]:
            if live in self.p.live_turnover:
                tot_turn = sum(self.p.live_turnover[live].values())
                tot_turn = max(tot_turn, 1e-6)
            else:
                tot_turn = 0.3
            self.pools[live] = self.p.NPP.get(live, 0.0) / max(tot_turn, 1e-6)

        U_ss = pd.Series(0.0, index=self.p.pool_names)
        for live, routes in self.p.live_turnover.items():
            for dst, frac in routes.items():
                U_ss[dst] += self.p.NPP.get(live, 0.0) * frac

        for pool in ["litter_foliage", "litter_woody", "dead_roots", "acrotelm_fast", "acrotelm_slow", "catotelm"]:
            dpar = self.p.decay[pool]
            a_k = applied_decay_rate(dpar.k_ref, mat_C, dpar.q10)
            self.pools[pool] = (U_ss.get(pool, 0.0) / a_k) if a_k > 0 else 0.0

        cat = "catotelm"
        dpar_cat = self.p.decay[cat]
        a_cat = applied_decay_rate(dpar_cat.k_ref, mat_C, dpar_cat.q10)
        U_cat = U_ss.get(cat, 0.0)
        if a_cat > 0:
            X_ss_cat = U_cat / a_cat
            ratio = (1.0 - math.exp(-a_cat * self.p.catotelm_years))
            self.pools[cat] = X_ss_cat * ratio
        return self.pools.copy()

    def simulate(self, years: int, mat_C_series: List[float], dc_series: List[float], seed: int = 123) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        records = []

        for t in range(years):
            mat = float(mat_C_series[t])
            dc = float(dc_series[t])
            WT = water_table_from_dc(dc, self.p.wt)

            live_to_dead = pd.Series(0.0, index=self.p.pool_names)
            for live, npp in self.p.NPP.items():
                self.pools[live] += npp
                for dst, frac in self.p.live_turnover.get(live, {}).items():
                    amount = self.pools[live] * frac
                    self.pools[live] -= amount
                    live_to_dead[dst] += amount

            atm_release = 0.0
            to_next_flux = pd.Series(0.0, index=self.p.pool_names)

            for pool in ["litter_foliage", "litter_woody", "dead_roots", "acrotelm_fast", "acrotelm_slow", "catotelm"]:
                dpar = self.p.decay[pool]
                a_k = applied_decay_rate(dpar.k_ref, mat, dpar.q10)
                D = a_k * self.pools[pool]
                route = self.p.routing[pool]
                to_atm = D * route.to_atm
                to_next = D * route.to_next
                atm_release += to_atm
                to_next_flux[pool] = to_next
                self.pools[pool] += live_to_dead.get(pool, 0.0)
                self.pools[pool] -= D

            chain = ["litter_foliage", "litter_woody", "dead_roots", "acrotelm_fast", "acrotelm_slow", "catotelm"]
            for i, pool in enumerate(chain[:-1]):
                nxt = chain[i+1]
                self.pools[nxt] += to_next_flux[pool]

            F_ch4_day = ch4_flux_gCm2_day(WT, self.p.ch4)
            F_ch4_yr = F_ch4_day * 365.0

            fire_event = False
            if self.p.fire.return_interval_yr > 0:
                p_fire = 1.0 / self.p.fire.return_interval_yr
                if rng.uniform() < p_fire:
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
                    peat_C = C_cum(depth, self.p.cdens)
                    for pool in ["acrotelm_fast", "acrotelm_slow"]:
                        take = min(self.pools[pool], peat_C * 0.5)
                        self.pools[pool] -= take
                        atm_release += take

            rec = {"year": t, "MAT_C": mat, "DC": dc, "WT_cm": WT, "atm_release": atm_release,
                   "CH4_gCm2_yr": F_ch4_yr, "fire": int(fire_event)}
            for name in self.p.pool_names:
                rec[f"pool_{name}"] = self.pools[name]
            records.append(rec)

        return pd.DataFrame.from_records(records)

# -------------------------------
# Defaults
# -------------------------------

def default_params() -> ModelParams:
    p = ModelParams()
    p.decay = {
        "litter_foliage": DecayParams(k_ref=0.8,  q10=Q10Params(Q10=2.0)),
        "litter_woody":   DecayParams(k_ref=0.2,  q10=Q10Params(Q10=2.0)),
        "dead_roots":     DecayParams(k_ref=0.5,  q10=Q10Params(Q10=2.0)),
        "acrotelm_fast":  DecayParams(k_ref=0.15, q10=Q10Params(Q10=2.0)),
        "acrotelm_slow":  DecayParams(k_ref=0.03, q10=Q10Params(Q10=2.0)),
        "catotelm":       DecayParams(k_ref=0.001,q10=Q10Params(Q10=2.0)),
    }
    p.routing = {
        "litter_foliage": PoolRouting(to_atm=0.8, to_next=0.2),
        "litter_woody":   PoolRouting(to_atm=0.6, to_next=0.4),
        "dead_roots":     PoolRouting(to_atm=0.7, to_next=0.3),
        "acrotelm_fast":  PoolRouting(to_atm=0.3, to_next=0.7),
        "acrotelm_slow":  PoolRouting(to_atm=0.1, to_next=0.9),
        "catotelm":       PoolRouting(to_atm=0.01,to_next=0.0),
    }
    p.live_turnover = {
        "live_trees": {"litter_foliage": 0.2, "litter_woody": 0.05, "dead_roots": 0.05},
        "live_shrubs": {"litter_foliage": 0.3, "dead_roots": 0.1},
        "live_sedge": {"litter_foliage": 0.5, "dead_roots": 0.2},
        "live_moss": {"litter_foliage": 0.7},
    }
    p.NPP = {"live_trees": 150.0, "live_shrubs": 90.0, "live_sedge": 120.0, "live_moss": 60.0}
    p.wt = WTParams(b_wc=-15.0)
    p.ch4 = CH4Params(F10_up=0.32, F10_down=2.6, WT_opt_cm=-10.0, F_CH4max_gCm2_day=0.12)
    p.fire = WildfireParams(return_interval_yr=150, combust_frac_live=0.8, combust_frac_litter=0.6, avg_burn_depth_cm=5.0)
    p.cdens = CarbonDensityParams(beta0=-3.0, beta1=0.03, beta2=-0.0002, gamma=60.0, delta=0.012)
    p.catotelm_years = 8000
    return p
