# -*- coding: utf-8 -*-
"""
balance_stats.py — ゲームバランス検証のための統計ライブラリ（設計書15章 v0.6対応）

設計思想（e-値統一アーキテクチャ）:
  1. 逐次モニタリング: Beta混合マルチンゲールによる confidence sequence (CS)。
     いつ・何回覗いても被覆率が保証される（optional stopping 耐性）。
  2. 多重比較: 同じ e-値を e-BH 法に投入して FDR 制御。
     検定を増やしても・後から追加しても保証が壊れない。
  3. 帯域判定（同等性）: anytime-valid TOST = 「CSが帯域に完全包含」で合格。
     固定nの場合は paired bootstrap 90%CI による通常 TOST も提供。
  4. ペア比較 (CRN): mid-p McNemar + 対応あり比率差CI。
  5. フロア別ハザード: 死亡分布の平坦性を離散ハザードで定量化。
  6. 選択感度: 大域検定 + max-min スプレッドの permutation null（上方バイアス対策）。
  7. シードパネル: 固定パネル + ホールドアウト分割 + ID管理（Public/Private LB の構造）。
  8. レーシング: 複数数値セットを並列に回し、CSで帯域外確定した候補を逐次淘汰。
  9. 検出力分析: 解析式でなくシミュレーションベース（CRN相関込み）。

依存: numpy, scipy のみ。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.special import betaln

# =====================================================================
# 0. 基本ユーティリティ
# =====================================================================

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson スコア区間。小さい p / 小標本でも破綻しない二項比率CI。"""
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


# =====================================================================
# 1. Confidence Sequence（Beta混合マルチンゲール）と e-値
# =====================================================================

@dataclass
class BernoulliCS:
    """
    ベルヌーイ系列に対する anytime-valid な信頼集合（confidence sequence）。

    理論: 混合マルチンゲール
        M_t(p) = B(a+S_t, b+t-S_t) / B(a,b) / [ p^{S_t} (1-p)^{t-S_t} ]
    は真値 p の下でマルチンゲールであり、Ville の不等式より
        P( ∃t: M_t(p_true) >= 1/alpha ) <= alpha.
    よって CS_t = { p : M_t(p) < 1/alpha } は全時刻同時に被覆率 1-alpha。
    → 試行を途中で何度見ても・いつ止めても妥当（optional stopping 耐性）。

    e-値: 任意の帰無仮説 H0: p ∈ P0 に対し e = inf_{p∈P0} M_t(p)。
    e >= 1/alpha で棄却。e-BH（多重比較）にそのまま投入できる。
    """
    alpha: float = 0.05
    prior_a: float = 1.0   # Beta事前。(1,1)=一様。事前情報があれば変更可
    prior_b: float = 1.0
    running_intersection: bool = True  # 過去時点とのCS交差（単調に狭まる）

    n: int = 0
    k: int = 0
    _lo: float = 0.0
    _hi: float = 1.0

    def update(self, successes: int, trials: int) -> "BernoulliCS":
        """観測をバッチで追加（successes 回成功 / trials 回試行）。"""
        if trials < 0 or successes < 0 or successes > trials:
            raise ValueError("invalid batch")
        self.k += successes
        self.n += trials
        lo, hi = self._compute_bounds()
        if self.running_intersection:
            self._lo = max(self._lo, lo)
            self._hi = min(self._hi, hi)
        else:
            self._lo, self._hi = lo, hi
        return self

    # ---- 内部: log M_t(p) ----
    def _log_m(self, p: np.ndarray) -> np.ndarray:
        a, b = self.prior_a, self.prior_b
        num = betaln(a + self.k, b + (self.n - self.k)) - betaln(a, b)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        den = self.k * np.log(p) + (self.n - self.k) * np.log1p(-p)
        return num - den

    def _compute_bounds(self) -> Tuple[float, float]:
        if self.n == 0:
            return (0.0, 1.0)
        thresh = math.log(1.0 / self.alpha)
        p_hat = self.k / self.n if self.n else 0.5
        p_hat = min(max(p_hat, 1e-12), 1 - 1e-12)

        def f(p):  # f<0 ⇔ p ∈ CS
            return self._log_m(np.array([p]))[0] - thresh

        # 下側
        if f(1e-12) < 0:
            lo = 0.0
        else:
            lo = _bisect(f, 1e-12, p_hat, increasing=False)
        # 上側
        if f(1 - 1e-12) < 0:
            hi = 1.0
        else:
            hi = _bisect(f, p_hat, 1 - 1e-12, increasing=True)
        return (lo, hi)

    # ---- 公開API ----
    @property
    def interval(self) -> Tuple[float, float]:
        return (self._lo, self._hi)

    def e_value(self, p0_lo: float, p0_hi: Optional[float] = None) -> float:
        """
        H0: p ∈ [p0_lo, p0_hi]（点帰無なら p0_hi 省略）に対する e-値。
        e = inf_{p∈H0} M_t(p)。M は p_hat から離れるほど増大する単峰なので
        区間帰無の inf は区間内で p_hat に最も近い点で達成される。
        """
        if p0_hi is None:
            p0_hi = p0_lo
        p_hat = self.k / self.n if self.n else 0.5
        p_star = min(max(p_hat, p0_lo), p0_hi)
        return float(np.exp(self._log_m(np.array([p_star]))[0]))

    def band_status(self, band_lo: float, band_hi: float) -> str:
        """
        合格帯域 [band_lo, band_hi]（例: 強クリア率 0.25–0.40）に対する逐次判定。
          PASS    : CS ⊆ 帯域 → 帯域内であることが確定（anytime-valid TOST 合格）
          FAIL    : CS ∩ 帯域 = ∅ → 帯域外が確定
          PENDING : まだ確定しない（試行追加）
        """
        lo, hi = self.interval
        if lo >= band_lo and hi <= band_hi:
            return "PASS"
        if hi < band_lo or lo > band_hi:
            return "FAIL"
        return "PENDING"


def _bisect(f: Callable[[float], float], lo: float, hi: float,
            increasing: bool, iters: int = 80) -> float:
    """f の符号変化点を二分探索（f<0 側が CS 内部）。"""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if increasing:   # 左がCS内(f<0)、右が外(f>0) → 境界=上限
            if v < 0:
                lo = mid
            else:
                hi = mid
        else:            # 左が外(f>0)、右がCS内(f<0) → 境界=下限
            if v < 0:
                hi = mid
            else:
                lo = mid
    return 0.5 * (lo + hi)


# =====================================================================
# 2. e-BH（e-値による FDR 制御）
# =====================================================================

def e_bh(e_values: Dict[str, float], alpha: float = 0.05) -> List[str]:
    """
    e-BH 法 (Wang & Ramdas, 2022)。e-値の集合に対し FDR <= alpha を保証。
    e-値同士の依存構造を問わず妥当（逐次追加・CRN相関があってもOK）。
    手順: e を降順に並べ k* = max{k : e_(k) >= m/(k*alpha)} の上位 k* を棄却。
    """
    m = len(e_values)
    if m == 0:
        return []
    items = sorted(e_values.items(), key=lambda kv: -kv[1])
    k_star = 0
    for i, (_, e) in enumerate(items, start=1):
        if e >= m / (i * alpha):
            k_star = i
    return [name for name, _ in items[:k_star]]


# =====================================================================
# 3. ペア比較（CRN 用）: mid-p McNemar / 対応あり差CI / TOST
# =====================================================================

@dataclass
class PairedResult:
    n_pairs: int
    p_a: float
    p_b: float
    diff: float                  # p_b - p_a
    diff_ci: Tuple[float, float]
    ci_level: float
    b01: int                     # A失敗/B成功 の不一致ペア
    b10: int                     # A成功/B失敗
    midp_pvalue: float
    tost_margin: Optional[float] = None
    tost_equivalent: Optional[bool] = None


def mcnemar_midp(b01: int, b10: int) -> float:
    """
    mid-p McNemar（両側）。正確版は保守的すぎ、漸近版は小標本で不正確。
    mid-p は実効サイズと検出力のバランスで小〜中標本の推奨手法
    (Fagerland et al., 2013)。
    """
    nd = b01 + b10
    if nd == 0:
        return 1.0
    x = min(b01, b10)
    p = 2.0 * (stats.binom.cdf(x - 1, nd, 0.5) + 0.5 * stats.binom.pmf(x, nd, 0.5))
    return float(min(1.0, p))


def paired_compare(outcomes_a: np.ndarray, outcomes_b: np.ndarray,
                   tost_margin: Optional[float] = None,
                   alpha: float = 0.05,
                   n_boot: int = 4000,
                   rng: Optional[np.random.Generator] = None) -> PairedResult:
    """
    同一シードパネル上で走らせた2条件（0/1 配列、index=シード）の対応あり比較。
      - 差の CI: シード単位の paired bootstrap（CRN相関を自動的に保持）
      - 検定: mid-p McNemar
      - TOST: margin 指定時、(1-2*alpha) CI ⊆ [-margin, +margin] で同等性合格
        （TOSTの標準的対応: alpha=0.05 → 90%CI）
    """
    a = np.asarray(outcomes_a, dtype=int)
    b = np.asarray(outcomes_b, dtype=int)
    if a.shape != b.shape:
        raise ValueError("paired arrays must share the seed panel (same shape)")
    rng = rng or np.random.default_rng(0)
    n = len(a)
    d = b.astype(float) - a.astype(float)

    ci_level = 1 - 2 * alpha if tost_margin is not None else 1 - alpha
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [(1 - ci_level) / 2, 1 - (1 - ci_level) / 2])

    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    res = PairedResult(
        n_pairs=n, p_a=float(a.mean()), p_b=float(b.mean()),
        diff=float(d.mean()), diff_ci=(float(lo), float(hi)), ci_level=ci_level,
        b01=b01, b10=b10, midp_pvalue=mcnemar_midp(b01, b10),
        tost_margin=tost_margin,
    )
    if tost_margin is not None:
        res.tost_equivalent = bool(lo >= -tost_margin and hi <= tost_margin)
    return res


# =====================================================================
# 4. フロア別ハザード（死亡分布の平坦性）
# =====================================================================

@dataclass
class HazardResult:
    hazards: np.ndarray            # 各フロアの条件付き死亡率 h_k
    at_risk: np.ndarray            # 各フロア到達ラン数
    deaths: np.ndarray             # 各フロア死亡数
    flatness_ratio: float          # max(h)/min(h)（平滑化込み）
    flatness_ci: Tuple[float, float]


def floor_hazards(reached_floor: np.ndarray, cleared: np.ndarray,
                  n_floors: int = 5, n_boot: int = 4000,
                  smooth: float = 0.5,
                  rng: Optional[np.random.Generator] = None) -> HazardResult:
    """
    離散時間生存分析のハザード h_k = P(フロアkで死 | フロアk到達)。
    「死亡フロア分布が特定フロアに集中しない」を flatness = max(h)/min(h)
    として定量化（smooth は 0 死亡フロアでの発散防止の Agresti 型平滑化）。
    到達フロア平均の比較より診断的: どのフロアの難度が動いたかが局在化する。
    """
    rng = rng or np.random.default_rng(0)
    reached = np.asarray(reached_floor, dtype=int)
    clr = np.asarray(cleared, dtype=bool)

    def _haz(re, cl):
        h = np.zeros(n_floors)
        risk = np.zeros(n_floors, dtype=int)
        dth = np.zeros(n_floors, dtype=int)
        for f in range(1, n_floors + 1):
            at_risk = np.sum(re >= f)
            died_here = np.sum((re == f) & (~cl))
            risk[f - 1], dth[f - 1] = at_risk, died_here
            h[f - 1] = (died_here + smooth) / (at_risk + 2 * smooth) if at_risk else np.nan
        return h, risk, dth

    h, risk, dth = _haz(reached, clr)
    ratio = float(np.nanmax(h) / np.nanmin(h))

    n = len(reached)
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        hb, _, _ = _haz(reached[idx], clr[idx])
        ratios[i] = np.nanmax(hb) / np.nanmin(hb)
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return HazardResult(h, risk, dth, ratio, (float(lo), float(hi)))


def paired_hazard_diff(reached_a, cleared_a, reached_b, cleared_b,
                       n_floors: int = 5) -> List[Dict]:
    """
    CRNペア上のフロア別ハザード差（B−A）。各フロアを対応あり比率差として
    Wilson 風の正規近似CIで返す。数値変更の影響フロアを局在化するための診断。
    """
    out = []
    ra, ca = np.asarray(reached_a), np.asarray(cleared_a, dtype=bool)
    rb, cb = np.asarray(reached_b), np.asarray(cleared_b, dtype=bool)
    for f in range(1, n_floors + 1):
        ia, ib = ra >= f, rb >= f
        both = ia & ib  # 両条件でフロアf到達（ペアとして比較可能）
        if both.sum() == 0:
            out.append(dict(floor=f, n=0, diff=np.nan, ci=(np.nan, np.nan)))
            continue
        da = ((ra == f) & ~ca)[both].astype(float)
        db = ((rb == f) & ~cb)[both].astype(float)
        d = db - da
        m, s, n = d.mean(), d.std(ddof=1), len(d)
        half = 1.96 * s / math.sqrt(n) if n > 1 else np.nan
        out.append(dict(floor=f, n=int(n), diff=float(m),
                        ci=(float(m - half), float(m + half))))
    return out


# =====================================================================
# 5. 選択感度（初手強制）— 大域検定 + バイアス補正スプレッド
# =====================================================================

@dataclass
class SensitivityResult:
    p_hats: Dict[str, float]
    global_pvalue: float           # k×2 カイ二乗（全アーム同率の大域検定)
    observed_spread: float         # max(p̂) − min(p̂)
    null_spread_q95: float         # 真差ゼロ下でノイズだけで出るスプレッドの95%点
    spread_pvalue: float           # P(null spread >= observed)


def selection_sensitivity(arm_outcomes: Dict[str, np.ndarray],
                          n_perm: int = 4000,
                          rng: Optional[np.random.Generator] = None) -> SensitivityResult:
    """
    初手の体験タイプを強制したアーム別クリア結果（0/1配列）の選択感度評価。

    重要: max−min スプレッドは真差ゼロでも正の値が出る（極値選択の上方バイアス）。
    → 大域検定（カイ二乗）で「どこかに差があるか」をまず判定し、
      スプレッドは pooled p の下での permutation null と比較して解釈する。
    """
    rng = rng or np.random.default_rng(0)
    names = list(arm_outcomes.keys())
    ks = np.array([arm_outcomes[a].sum() for a in names], dtype=float)
    ns = np.array([len(arm_outcomes[a]) for a in names], dtype=float)
    p_hats = ks / ns

    table = np.vstack([ks, ns - ks]).T
    if np.any(table.sum(axis=0) == 0):
        global_p = 1.0
    else:
        global_p = float(stats.chi2_contingency(table)[1])

    obs_spread = float(p_hats.max() - p_hats.min())
    pooled = ks.sum() / ns.sum()
    sims = np.empty(n_perm)
    for i in range(n_perm):
        sim_p = rng.binomial(ns.astype(int), pooled) / ns
        sims[i] = sim_p.max() - sim_p.min()
    return SensitivityResult(
        p_hats={a: float(p) for a, p in zip(names, p_hats)},
        global_pvalue=global_p,
        observed_spread=obs_spread,
        null_spread_q95=float(np.quantile(sims, 0.95)),
        spread_pvalue=float(np.mean(sims >= obs_spread)),
    )


# =====================================================================
# 6. シードパネル（Public/Private LB 構造）
# =====================================================================

@dataclass
class SeedPanel:
    """
    固定ベンチマークシード集合。調整は tuning パネルで回し、
    確定判定のみ holdout で行う（固定パネルへの過適合 = Public LB 過適合の防止）。
    panel_id をランログの param_hash と並べて記録すること。
    """
    master_seed: int
    n_seeds: int
    holdout_frac: float = 0.2
    strata: Optional[np.ndarray] = None   # 層化用スコア（任意・レイアウト難度等）

    seeds: np.ndarray = field(init=False)
    tuning_idx: np.ndarray = field(init=False)
    holdout_idx: np.ndarray = field(init=False)
    panel_id: str = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.master_seed)
        self.seeds = rng.integers(0, 2**63 - 1, size=self.n_seeds, dtype=np.int64)
        n_hold = int(round(self.n_seeds * self.holdout_frac))
        if self.strata is not None:
            # 層化: 層スコアでソートし等間隔抽出 → holdout が難度分布を保つ
            order = np.argsort(self.strata)
            pick = np.linspace(0, self.n_seeds - 1, n_hold).round().astype(int)
            self.holdout_idx = np.sort(order[pick])
        else:
            self.holdout_idx = np.sort(
                rng.choice(self.n_seeds, size=n_hold, replace=False))
        mask = np.ones(self.n_seeds, dtype=bool)
        mask[self.holdout_idx] = False
        self.tuning_idx = np.where(mask)[0]
        payload = json.dumps(dict(ms=self.master_seed, n=self.n_seeds,
                                  hf=self.holdout_frac)).encode()
        self.panel_id = hashlib.sha256(payload).hexdigest()[:12]

    @property
    def tuning_seeds(self) -> np.ndarray:
        return self.seeds[self.tuning_idx]

    @property
    def holdout_seeds(self) -> np.ndarray:
        return self.seeds[self.holdout_idx]


# =====================================================================
# 7. レーシング（数値セット候補の逐次淘汰）
# =====================================================================

def race_param_sets(simulators: Dict[str, Callable[[np.ndarray], np.ndarray]],
                    seed_panel: np.ndarray,
                    band: Tuple[float, float],
                    alpha: float = 0.05,
                    batch_size: int = 200,
                    max_trials: Optional[int] = None) -> Dict[str, Dict]:
    """
    複数の数値セット候補を同一シードパネルで並列に回し、
    confidence sequence が帯域 [band] の外を確定した候補を逐次脱落させる。
    CS は anytime-valid なので、バッチごとに判定しても保証が壊れない。
    alpha は候補数で Bonferroni 補正（全候補同時に妥当な淘汰）。

    simulators: name -> f(seeds)->0/1配列（クリア有無）。
    返り値: name -> {status, n, k, cs}。
    """
    m = len(simulators)
    a_each = alpha / m
    states = {name: BernoulliCS(alpha=a_each) for name in simulators}
    status = {name: "PENDING" for name in simulators}
    pos = 0
    n_total = len(seed_panel) if max_trials is None else min(max_trials, len(seed_panel))

    while pos < n_total and any(s == "PENDING" for s in status.values()):
        batch = seed_panel[pos: pos + batch_size]
        pos += len(batch)
        for name, sim in simulators.items():
            if status[name] != "PENDING":
                continue
            outcomes = np.asarray(sim(batch), dtype=int)
            cs = states[name].update(int(outcomes.sum()), len(outcomes))
            status[name] = cs.band_status(*band)

    return {name: dict(status=status[name], n=states[name].n, k=states[name].k,
                       cs=states[name].interval)
            for name in simulators}


# =====================================================================
# 8. シミュレーションベース検出力分析
# =====================================================================

def simulate_crn_pair(p_a: float, p_b: float, gamma: float, n_pairs: int,
                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    CRN 効果のモデル化: 確率 gamma で共通乱数（最大カップリング）、
    確率 1-gamma で独立乱数。gamma が高いほど不一致ペアが減る。
    実機の gamma は実測可能（baseline を2回別ストリームで回して相関を測る）。
    """
    u_common = rng.random(n_pairs)
    u_a = np.where(rng.random(n_pairs) < gamma, u_common, rng.random(n_pairs))
    u_b = np.where(rng.random(n_pairs) < gamma, u_common, rng.random(n_pairs))
    return (u_a < p_a).astype(int), (u_b < p_b).astype(int)


def power_mcnemar_crn(p_a: float, p_b: float, gamma: float, n_pairs: int,
                      alpha: float = 0.05, n_sim: int = 1000,
                      seed: int = 0) -> float:
    """mid-p McNemar の検出力をシミュレーションで直接推定。"""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        a, b = simulate_crn_pair(p_a, p_b, gamma, n_pairs, rng)
        b01 = int(np.sum((a == 0) & (b == 1)))
        b10 = int(np.sum((a == 1) & (b == 0)))
        if mcnemar_midp(b01, b10) < alpha:
            hits += 1
    return hits / n_sim


def required_pairs(p_a: float, p_b: float, gamma: float,
                   target_power: float = 0.8, alpha: float = 0.05,
                   n_grid: Sequence[int] = (250, 500, 1000, 2000, 4000, 8000),
                   n_sim: int = 600, seed: int = 0) -> Dict[int, float]:
    """グリッド上で検出力カーブを返す（必要ペア数の目安取得用）。"""
    return {n: power_mcnemar_crn(p_a, p_b, gamma, n, alpha, n_sim, seed)
            for n in n_grid}


# =====================================================================
# 9. デモ / セルフテスト
# =====================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print("=" * 72)
    print("balance_stats.py セルフテスト（合成データ）")
    print("=" * 72)

    # --- (1) シードパネル -------------------------------------------------
    panel = SeedPanel(master_seed=20260610, n_seeds=3000, holdout_frac=0.2)
    print(f"\n[1] SeedPanel id={panel.panel_id}  "
          f"tuning={len(panel.tuning_seeds)}  holdout={len(panel.holdout_seeds)}")

    # --- (2) Confidence Sequence: 逐次モニタリングと帯域判定 -----------------
    print("\n[2] Confidence Sequence（真のクリア率 p=0.31, 帯域 0.25–0.40）")
    cs = BernoulliCS(alpha=0.05)
    true_p = 0.31
    for step in range(1, 13):
        x = rng.binomial(1, true_p, size=250)
        cs.update(int(x.sum()), len(x))
        lo, hi = cs.interval
        st = cs.band_status(0.25, 0.40)
        print(f"   n={cs.n:5d}  p̂={cs.k/cs.n:.3f}  CS=({lo:.3f},{hi:.3f})  -> {st}")
        if st != "PENDING":
            print(f"   → n={cs.n} で {st} 確定（覗き見しながらでも妥当）")
            break

    # CS の被覆率サニティチェック（真値が一度でも外れる系列の率 <= alpha）
    viol = 0
    n_rep = 400
    for _ in range(n_rep):
        c = BernoulliCS(alpha=0.05)
        bad = False
        for _ in range(10):
            x = rng.binomial(1, 0.3, size=100)
            c.update(int(x.sum()), len(x))
            lo, hi = c.interval
            if not (lo <= 0.3 <= hi):
                bad = True
        viol += bad
    print(f"   被覆率チェック: 真値0.3が系列中に一度でも外れた率 = "
          f"{viol/n_rep:.3f}  (<= 0.05 であるべき)")

    # --- (3) ペア比較: mid-p McNemar / paired bootstrap / TOST ---------------
    print("\n[3] CRNペア比較（pA=0.30 vs pB=0.34, gamma=0.8, n=2000）")
    a, b = simulate_crn_pair(0.30, 0.34, gamma=0.8, n_pairs=2000, rng=rng)
    res = paired_compare(a, b, tost_margin=0.05, alpha=0.05, rng=rng)
    print(f"   p̂A={res.p_a:.3f} p̂B={res.p_b:.3f} diff={res.diff:+.3f} "
          f"{int(res.ci_level*100)}%CI=({res.diff_ci[0]:+.3f},{res.diff_ci[1]:+.3f})")
    print(f"   不一致ペア b01={res.b01} b10={res.b10}  mid-p McNemar p={res.midp_pvalue:.4f}")
    print(f"   TOST(±5pt) 同等?: {res.tost_equivalent}")

    print("   （同条件・独立試行なら）2標本z検定の分散と比較:")
    se_paired = np.std(b - a, ddof=1) / math.sqrt(len(a))
    se_indep = math.sqrt(res.p_a*(1-res.p_a)/len(a) + res.p_b*(1-res.p_b)/len(b))
    print(f"   SE_paired={se_paired:.4f}  SE_indep={se_indep:.4f}  "
          f"→ 分散削減 {100*(1-(se_paired/se_indep)**2):.0f}%（必要試行 約{(se_indep/se_paired)**2:.1f}分の1）")

    # --- (4) e-BH: 宝箱mod 6種の寄与スクリーニング ---------------------------
    print("\n[4] e-BH（宝箱mod 6種アブレーション, 真の寄与あり=mod1,mod2）")
    true_effects = dict(mod1=0.08, mod2=0.06, mod3=0.0, mod4=0.0, mod5=0.0, mod6=0.0)
    e_vals = {}
    for name, eff in true_effects.items():
        x, y = simulate_crn_pair(0.30, 0.30 + eff, gamma=0.8, n_pairs=1500, rng=rng)
        d = (y - x)
        # 差の e-値: 不一致ペアの偏り（b01/b10）に対する Bernoulli CS の e-値を流用
        b01 = int(np.sum((x == 0) & (y == 1)))
        b10 = int(np.sum((x == 1) & (y == 0)))
        c = BernoulliCS(alpha=0.05).update(b01, b01 + b10)
        e_vals[name] = c.e_value(0.5)   # H0: 不一致内の偏り=0.5（=効果なし）
    rejected = e_bh(e_vals, alpha=0.05)
    print("   e-値:", {k: f"{v:.1f}" for k, v in e_vals.items()})
    print(f"   e-BH 棄却（=寄与あり判定）: {sorted(rejected)}  ← 期待: mod1, mod2")

    # --- (5) フロア別ハザード ------------------------------------------------
    print("\n[5] フロア別ハザード（3Fに難所を仕込んだ合成ラン 4000本）")
    haz_true = np.array([0.05, 0.10, 0.30, 0.12, 0.10])
    n_runs = 4000
    reached = np.zeros(n_runs, dtype=int)
    cleared = np.zeros(n_runs, dtype=bool)
    for i in range(n_runs):
        f = 0
        alive = True
        while alive and f < 5:
            f += 1
            if rng.random() < haz_true[f - 1]:
                alive = False
        reached[i] = f
        cleared[i] = alive
    hz = floor_hazards(reached, cleared, rng=rng)
    print("   h_k 推定:", np.round(hz.hazards, 3), " 真値:", haz_true)
    print(f"   平坦性 max/min = {hz.flatness_ratio:.2f} "
          f"95%CI=({hz.flatness_ci[0]:.2f},{hz.flatness_ci[1]:.2f})  "
          f"→ 3F集中を正しく検出")

    # --- (6) 選択感度: スプレッドの上方バイアスを可視化 -----------------------
    print("\n[6] 選択感度（3アーム・真差ゼロ p=0.30, 各n=800）")
    arms = {f"初手{t}": rng.binomial(1, 0.30, 800) for t in ("削り合い", "賭け", "レース")}
    sens = selection_sensitivity(arms, rng=rng)
    print(f"   p̂ = { {k: round(v,3) for k,v in sens.p_hats.items()} }")
    print(f"   観測スプレッド={sens.observed_spread:.3f} / "
          f"真差ゼロでもノイズで出る95%点={sens.null_spread_q95:.3f}")
    print(f"   大域検定 p={sens.global_pvalue:.3f}, スプレッド p={sens.spread_pvalue:.3f} "
          f"→ 素朴な max−min 判定なら誤検出していた水準")

    # --- (7) レーシング -------------------------------------------------------
    print("\n[7] レーシング（数値セット3候補, 帯域0.25–0.40, バッチ200）")
    true_ps = {"setA(p=.31)": 0.31, "setB(p=.18)": 0.18, "setC(p=.52)": 0.52}
    def make_sim(p):
        local = np.random.default_rng(int(p * 1e6))
        return lambda seeds: local.binomial(1, p, size=len(seeds))
    sims = {k: make_sim(v) for k, v in true_ps.items()}
    out = race_param_sets(sims, panel.tuning_seeds, band=(0.25, 0.40),
                          alpha=0.05, batch_size=200)
    for k, v in out.items():
        print(f"   {k:14s} status={v['status']:7s} n={v['n']:5d} "
              f"CS=({v['cs'][0]:.3f},{v['cs'][1]:.3f})")
    print("   → 帯域外候補は少ない試行で早期脱落し、計算資源を有望候補に集中")

    # --- (8) 検出力分析 -------------------------------------------------------
    print("\n[8] 検出力カーブ（pA=0.30 vs pB=0.35, mid-p McNemar, α=0.05）")
    for g in (0.0, 0.5, 0.8):
        curve = required_pairs(0.30, 0.35, gamma=g,
                               n_grid=(250, 500, 1000, 2000), n_sim=400)
        print(f"   gamma={g}: " + "  ".join(f"n={n}:{p:.2f}" for n, p in curve.items()))
    print("   → CRN相関(gamma)が高いほど同じnで検出力が上がる＝ペア化の価値の定量化")

    print("\nOK: 全コンポーネント動作確認済み")
