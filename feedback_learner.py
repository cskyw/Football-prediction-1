"""
feedback_learner.py
====================
实盘反馈学习模块
功能：
  1. 预测时自动保存特征+预测结果到 prediction_log.csv
  2. 输入真实赛果后，评估模型表现 + 触发增量学习
  3. 可视化追踪累计准确率趋势

使用方式：
  - 在 EV_prediction_3.py 的 predict() 函数末尾调用 save_prediction_record(...)
  - 赛后运行本脚本：python feedback_learner.py，输入赛果即可
"""

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ========================
# 配置
# ========================
LOG_FILE = "prediction_log.csv"  # 预测记录文件

LEAGUES = {
    "1": {"name": "英超",  "model": "premier_model.pkl"},
    "2": {"name": "英冠",  "model": "championship_model.pkl"},
    "3": {"name": "西甲",  "model": "laliga2_model.pkl"},
    "4": {"name": "意甲",  "model": "italy2_model.pkl"},
    "5": {"name": "葡超",  "model": "portugal2_model.pkl"},
}

FEATURES = [
    "strong_win_odds", "draw_odds", "weak_win_odds",
    "odds_gap", "odds_ratio", "p_strong", "strong_is_home",
    "strong_odds_move", "weak_odds_move", "strong_move_pct", "weak_move_pct",
    "strong_avg_shots", "weak_attack_score", "weak_defense_score",
    "ps_p_strong", "odds_market_diff", "strong_avg_corners", "weak_avg_corners",
]

# ========================
# 1. 保存预测记录（在 EV_prediction_3.py 中调用）
# ========================
def save_prediction_record(
    league_name, home_team, away_team, match_date,
    features: dict,
    prob_upset: float,
    prob_strong: float,
    ev_strong: float,
    recommendation: str,
):
    """
    在 EV_prediction_3.py 的 predict() 函数里调用此函数，
    把本次预测的所有信息存入 prediction_log.csv。
    """
    record = {
        "log_time":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "league":        league_name,
        "match_date":    str(match_date)[:10],
        "home_team":     home_team,
        "away_team":     away_team,
        "prob_upset":    round(prob_upset, 4),
        "prob_strong":   round(prob_strong, 4),
        "ev_strong":     round(ev_strong, 4),
        "recommendation": recommendation,
        # 真实赛果（待填入）
        "fthg":          np.nan,   # 主队进球
        "ftag":          np.nan,   # 客队进球
        "upset_actual":  np.nan,   # 1=冷门发生，0=强队赢
        "result_filled": 0,        # 是否已填入赛果
        "is_qualified": 1 if (features.get("strong_win_odds", 99) <= 1.8 and
                              features.get("weak_win_odds", 0) >= 4) else 0,
    }
    # 把特征也存进去，方便后续增量训练
    for f in FEATURES:
        record[f] = features.get(f, np.nan)

    df_log = _load_log()
    new_row = pd.DataFrame([record])
    df_log = pd.concat([df_log, new_row], ignore_index=True)
    df_log.to_csv(LOG_FILE, index=False)
    print(f"  ✅ 预测记录已保存 → {LOG_FILE}")


# ========================
# 2. 填入赛果
# ========================
def fill_results():
    """交互式填入昨天/指定日期比赛的真实赛果"""
    df_log = _load_log()
    if df_log.empty:
        print("暂无预测记录。")
        return

    # 找未填赛果的记录
    pending = df_log[df_log["result_filled"] == 0].copy()
    if pending.empty:
        print("✅ 所有预测记录已填写赛果，无待处理。")
        return

    print(f"\n共有 {len(pending)} 场比赛待填入赛果：\n")
    print(pending[["match_date", "league", "home_team", "away_team", "prob_upset", "recommendation"]].to_string(index=True))

    print("\n请逐一输入赛果（直接回车跳过该场）：")
    for idx, row in pending.iterrows():
        print(f"\n  [{idx}] {row['match_date']}  {row['home_team']} vs {row['away_team']}  ({row['league']})")
        s = input("    主队进球数 (回车跳过): ").strip()
        if s == "":
            continue
        try:
            fthg = int(s)
            ftag = int(input("    客队进球数: ").strip())
        except ValueError:
            print("    输入无效，跳过。")
            continue

        # 判断强队是否主场
        strong_is_home = int(df_log.at[idx, "strong_is_home"])
        if strong_is_home:
            upset_actual = 1 if fthg <= ftag else 0
        else:
            upset_actual = 1 if ftag <= fthg else 0

        df_log.at[idx, "fthg"]          = fthg
        df_log.at[idx, "ftag"]          = ftag
        df_log.at[idx, "upset_actual"]  = upset_actual
        df_log.at[idx, "result_filled"] = 1

        result_str = "冷门✅" if upset_actual else "强队赢❌"
        pred_str   = "预测冷门" if row["prob_upset"] >= 0.5 else "预测强队赢"
        correct    = (upset_actual == 1 and row["prob_upset"] >= 0.5) or \
                     (upset_actual == 0 and row["prob_upset"] < 0.5)
        print(f"    赛果: {fthg}:{ftag}  →  {result_str}  |  {pred_str}  {'✔ 命中' if correct else '✘ 未中'}")

    df_log.to_csv(LOG_FILE, index=False)
    print(f"\n✅ 赛果已保存至 {LOG_FILE}")
    return df_log




# ========================
# 3. 可视化追踪
# ========================
def plot_performance(league_name: str = None):
    df_log = _load_log()
    df_filled = df_log[df_log["result_filled"] == 1].copy()
    if league_name:
        df_filled = df_filled[df_filled["league"] == league_name]
    if df_filled.empty:
        print("暂无已填赛果的记录。")
        return

    df_filled = df_filled.sort_values("match_date").reset_index(drop=True)
    title_suffix = f" — {league_name}" if league_name else ""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for subset, label, color in [
        (df_filled[df_filled["is_qualified"] == 1], "符合赔率条件", "#2196F3"),
        (df_filled[df_filled["is_qualified"] == 0], "不符合赔率条件", "#FF9800"),
    ]:
        if subset.empty:
            continue
        subset = subset.reset_index(drop=True)
        subset["correct"]   = ((subset["prob_upset"] >= 0.5).astype(int) == subset["upset_actual"]).astype(int)
        subset["cum_acc"]   = subset["correct"].expanding().mean()
        pred_upset_mask     = subset["prob_upset"] >= 0.5
        subset["cum_upset_hit"] = (
            ((pred_upset_mask) & (subset["upset_actual"] == 1))
            .expanding().sum() /
            pred_upset_mask.expanding().sum().replace(0, np.nan)
        )

        axes[0].plot(subset.index + 1, subset["cum_acc"] * 100,
                     color=color, linewidth=2, marker="o", markersize=4, label=label)
        axes[1].plot(subset.index + 1, subset["cum_upset_hit"] * 100,
                     color=color, linewidth=2, marker="s", markersize=4, label=label)

    axes[0].axhline(50, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title(f"累计预测准确率{title_suffix}", fontsize=14)
    axes[0].set_ylabel("准确率 (%)")
    axes[0].set_xlabel("场次")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 100)

    axes[1].axhline(30, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title(f"冷门命中率{title_suffix}", fontsize=14)
    axes[1].set_ylabel("命中率 (%)")
    axes[1].set_xlabel("场次")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("prediction_performance.png", dpi=150)
    plt.show()
    print("图表已保存 → prediction_performance.png")


# ========================
# 4. 打印统计摘要
# ========================
def print_summary(league_name: str = None):
    df_log = _load_log()
    df_filled = df_log[df_log["result_filled"] == 1].copy()
    if league_name:
        df_filled = df_filled[df_filled["league"] == league_name]
    if df_filled.empty:
        print("暂无数据。")
        return

    def _calc_stats(df, label):
        if df.empty:
            print(f"\n  【{label}】暂无数据")
            return
        total      = len(df)
        upset_real = df["upset_actual"].sum()
        pred_upset = (df["prob_upset"] >= 0.5).sum()
        correct    = ((df["prob_upset"] >= 0.5).astype(int) == df["upset_actual"]).sum()

        predicted_upset_df = df[df["prob_upset"] >= 0.5]
        upset_hit_rate = predicted_upset_df["upset_actual"].mean() if len(predicted_upset_df) > 0 else np.nan

        print(f"\n  【{label}】")
        print(f"  总预测场次:          {total}")
        print(f"  实际发生冷门:        {int(upset_real)}  ({upset_real/total*100:.1f}%)")
        print(f"  模型预测为冷门:      {int(pred_upset)}")
        print(f"  总体预测准确:        {int(correct)}  ({correct/total*100:.1f}%)")
        if not np.isnan(upset_hit_rate):
            print(f"  冷门命中率:          {upset_hit_rate*100:.1f}%  (预测冷门时实际发生冷门)")
        else:
            print(f"  冷门命中率:          暂无预测冷门的场次")

    title = f"— {league_name}" if league_name else ""
    print("\n" + "="*45)
    print(f"  实盘追踪摘要  {title}")
    print("="*45)

    # 符合赔率条件
    df_qualified   = df_filled[df_filled["is_qualified"] == 1]
    # 不符合赔率条件
    df_unqualified = df_filled[df_filled["is_qualified"] == 0]

    _calc_stats(df_qualified,   "符合赔率条件（强队≤1.8 / 弱队≥4）")
    _calc_stats(df_unqualified, "不符合赔率条件（仅供参考）")

    print("\n" + "="*45)


# ========================
# 内部工具
# ========================
def _load_log() -> pd.DataFrame:
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df["trained"] = df["trained"].fillna(0)
        # 兼容旧记录：如果没有is_qualified列，默认全部为1
        if "is_qualified" not in df.columns:
            df["is_qualified"] = 1
        return df
    cols = ["log_time", "league", "match_date", "home_team", "away_team",
            "prob_upset", "prob_strong", "ev_strong", "recommendation",
            "fthg", "ftag", "upset_actual", "result_filled", "trained",
            "is_qualified"] + FEATURES
    return pd.DataFrame(columns=cols)

# ========================
# 主菜单（独立运行）
# ========================
if __name__ == "__main__":
    print("="*45)
    print("   实盘反馈学习系统")
    print("="*45)
    print("1. 填入赛果")
    print("2. 查看预测统计摘要")
    print("3. 可视化追踪图表")
    print("0. 退出")

    while True:
        choice = input("\n请选择操作: ").strip()

        if choice == "1":
            fill_results()

        elif choice == "2":
            print("\n查看全部联赛输入 0，或选择具体联赛：")
            for k, v in LEAGUES.items():
                print(f"  {k}. {v['name']}")
            lc = input("输入编号（0=全部）: ").strip()
            ln = LEAGUES[lc]["name"] if lc in LEAGUES else None
            print_summary(ln)

        elif choice == "3":
            print("\n查看全部联赛输入 0，或选择具体联赛：")
            for k, v in LEAGUES.items():
                print(f"  {k}. {v['name']}")
            lc = input("输入编号（0=全部）: ").strip()
            ln = LEAGUES[lc]["name"] if lc in LEAGUES else None
            plot_performance(ln)

        elif choice == "0":
            print("退出。")
            break
        else:
            print("无效选择，请重试。")