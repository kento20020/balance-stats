# balance_stats.py 統合ガイド（設計書15章 v0.6 対応）

`phase12_harness.py` / `balance_analysis.py` に `balance_stats.py` を組み込み、
**e-値統一アーキテクチャ**でバランス検証を運用するための手順。

---

## 0. アーキテクチャの全体像

```
                    ┌─ 覗き見・逐次追加 ──→ Confidence Sequence (Ville)
e-値（1つの通貨）──┤
                    └─ 多重比較（FDR）  ──→ e-BH
```

| 旧（v0.5まで） | 新（v0.6） | 解決する問題 |
|---|---|---|
| Wilson CI を随時確認 | **BernoulliCS.band_status()** | optional stopping（覗き見で偽合格率が膨張） |
| 「±5pt以内」を目視判定 | **TOST**（paired_compare の tost_margin）/ CS包含 | 「有意でない＝同等」の誤謬 |
| 正確 McNemar | **mid-p McNemar** | 保守性による検出力ロス |
| 検定をα=0.05で乱発 | **e-BH** | 多重比較（数値セット1つで20-30検定走る） |
| 死亡フロア分布を目視 | **floor_hazards**（平坦性 max/min） | 集中度の定量化・影響フロアの局在化 |
| 初手差の max−min | **selection_sensitivity**（大域検定+permutation null） | スプレッドの上方バイアス |
| 毎回シード新規生成 | **SeedPanel**（tuning/holdout 分割） | 固定パネル過適合（Public LB 過適合と同型） |
| 全候補を同じ試行数 | **race_param_sets** | 帯域外候補への計算資源の浪費 |
| 解析式で必要n見積り | **power_mcnemar_crn**（シミュレーションベース） | CRN相関込みの正確な検出力 |

---

## 1. ハーネス側に必要な変更（3点のみ）

### 1-1. シミュレータの関数化
```python
# phase12_harness.py 側
def run_batch(seeds: np.ndarray, config: dict, policy: str) -> dict:
    """seeds の各シードで1ラン実行し、結果をまとめて返す"""
    cleared, reached, first_pick = [], [], []
    for s in seeds:
        r = run_one(seed=int(s), config=config, policy=policy)
        cleared.append(r.cleared)        # bool
        reached.append(r.reached_floor)  # 1..5
        first_pick.append(r.first_pick_type)  # "削り合い" 等
    return dict(cleared=np.array(cleared, dtype=int),
                reached=np.array(reached, dtype=int),
                first_pick=np.array(first_pick))
```

### 1-2. ランログへの ID 追記
既存の `param_hash` に加えて `panel_id`（SeedPanel.panel_id）と
`policy_name` を全ログに記録する。「どの数値・どのシード集合・どのbotで何%か」
を後から完全に追跡できるようにする。

### 1-3. CRN 相関 γ の実測
```python
from balance_stats import simulate_crn_pair  # 参考実装
# 実測: 同一 config を別 master_seed 由来の context_key で2回回し、
# クリア結果の一致率から γ を逆算（γ ≈ 2*P(一致) - 1 を初期近似に）
# → power_mcnemar_crn(γ) に入れて必要ペア数を見積もる
```

---

## 2. 運用ワークフロー（15.7 の置き換え）

### フェーズ0: 検出力分析（数値投入前に1回）
```python
from balance_stats import required_pairs, SeedPanel

panel = SeedPanel(master_seed=20260610, n_seeds=5000, holdout_frac=0.2)
# γ は 1-3 の実測値を使用
print(required_pairs(p_a=0.30, p_b=0.35, gamma=0.8))
# → 検出したい最小差に対する必要ペア数を確定し、パネルサイズを決める
```

### フェーズ1-2: 調整ループ（tuning パネル上・覗き見OK）
```python
from balance_stats import race_param_sets

results = race_param_sets(
    simulators={name: (lambda seeds, c=cfg: run_batch(seeds, c, "strong")["cleared"])
                for name, cfg in candidate_configs.items()},
    seed_panel=panel.tuning_seeds,
    band=(0.25, 0.40),       # 強クリア率の確定帯域
    alpha=0.05, batch_size=200)
# PASS した候補のみフェーズ2の併走指標（スキル幅・選択感度）へ進める
```

併走指標（PASS 候補に対して）:
```python
from balance_stats import paired_compare, selection_sensitivity, floor_hazards

# スキル幅: 強 vs random を同一シードでペア化（分散が大幅減）
strong = run_batch(panel.tuning_seeds, cfg, "strong")
randm  = run_batch(panel.tuning_seeds, cfg, "random")
skill = paired_compare(randm["cleared"], strong["cleared"])
# 合格条件: diff の CI が 0 をまたがない、かつ十分大きい

# 選択感度: 初手強制アーム別
arms = {t: run_batch(panel.tuning_seeds, cfg, f"force_first:{t}")["cleared"]
        for t in ("削り合い", "賭け", "レース", "ずれ")}
sens = selection_sensitivity(arms)
# 合格条件: global_pvalue < 0.05 かつ observed_spread > null_spread_q95
# （素朴な max−min 判定は禁止: 真差ゼロでもノイズで正の値が出る）

# 死亡分布: ハザード平坦性
hz = floor_hazards(strong["reached"], strong["cleared"].astype(bool))
# 合格条件（暫定）: flatness_ci 上限 < 3.0 程度から実測調整
```

### フェーズ3: アブレーション（sink ROI / 宝箱mod 寄与）
```python
from balance_stats import BernoulliCS, e_bh

e_vals = {}
for mod_name in all_mods:
    base = run_batch(panel.tuning_seeds, cfg, "strong")["cleared"]
    abl  = run_batch(panel.tuning_seeds, cfg_without(mod_name), "strong")["cleared"]
    b01 = int(((base == 0) & (abl == 1)).sum())
    b10 = int(((base == 1) & (abl == 0)).sum())
    e_vals[mod_name] = BernoulliCS().update(b01, b01 + b10).e_value(0.5)

significant = e_bh(e_vals, alpha=0.05)   # FDR<=5% で「寄与あり」確定
# Δ≈0 かつ低使用率 → 死にメカニクス / Δ突出 → 事実上必須（選択消失）
# の判定は paired_compare の diff_ci で効果量を併記して行う
```

### フェーズ4: 確定判定（holdout・1回限り）
```python
# tuning で PASS した最終候補のみ、未使用の holdout_seeds で再走。
final = run_batch(panel.holdout_seeds, best_cfg, "strong")
cs = BernoulliCS(alpha=0.05).update(int(final["cleared"].sum()),
                                    len(final["cleared"]))
assert cs.band_status(0.25, 0.40) == "PASS"   # これが正式合格
# holdout は使い捨て: 不合格なら新パネルを切り直す（過適合防止の鉄則）
```

### CI（継続的インテグレーション）の2層化
- **第1層（毎コミット）**: シード固定の決定論スナップショット厳密一致。
  乱数アーキテクチャの回帰はここで全て捕まる。統計不要。
- **第2層（数値セット変更時のみ）**: 上記フェーズ1-3を tuning パネルで実行。
  e-値で判定するため、テストを何度走らせても保証が壊れない。

---

## 3. 設計書 15.3 の書き換え案（合格条件表）

| 指標 | 目標 | 判定方法（新規列） |
|---|---|---|
| 強クリア率 | 25-40% | **CS が帯域に完全包含（anytime-valid）** |
| ランダムクリア率 | 1-5% | 同上 |
| スキル幅 | CIが0をまたがず十分大 | **CRNペア + paired bootstrap CI** |
| 選択感度 | 有意な振れ幅 | **大域カイ二乗 + permutation null 超過** |
| 初手別クリア率差 | ±5pt | **TOST（90%CIが帯域内）** |
| 死亡フロア分布 | 集中しない | **ハザード平坦性 max/min の CI 上限** |
| 単一mod寄与 | ±8pt | **TOST + e-BH（多数modの同時スクリーニング）** |
