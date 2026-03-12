import numpy as np
import pandas as pd
import os
import joblib
from dataloader import load_multiple_seasons


LEAGUES = {
    "1": {"name": "英超",  "folder": "data_Premier",       "model": "premier_model.pkl"},
    "2": {"name": "英冠",  "folder": "data_Championship",  "model": "championship_model_sklearn.pkl"},
    "3": {"name": "西甲",  "folder": "data_laliga",        "model": "laliga2_model.pkl"},
    "4": {"name": "意甲",  "folder": "data_italy",         "model": "italy2_model.pkl"},
    "5": {"name": "葡超",  "folder": "data_portugal",      "model": "portugal2_model.pkl"},
}

FEATURES = [
    "strong_win_odds",
    "draw_odds",
    "weak_win_odds",
    "odds_gap",
    "odds_ratio",
    "p_strong",
    "strong_is_home",
    "strong_odds_move",
    "weak_odds_move",
    "strong_move_pct",
    "weak_move_pct",
    "strong_avg_shots",
    "weak_attack_score",
    "weak_defense_score",
    "ps_p_strong",
    "odds_market_diff",
    "strong_avg_corners",
    "weak_avg_corners",
]

# =========================
# 启动时选择联赛
# =========================
print("="*45)
print("       冷门预测系统")
print("="*45)
print("\n请选择联赛：")
for k, v in LEAGUES.items():
    print(f"  {k}. {v['name']}")

while True:
    choice = os.environ.get("LEAGUE_CHOICE", "2")
    if choice in LEAGUES:
        break
    print("  无效编号，请重新输入！")

league_cfg  = LEAGUES[choice]
DATA_FOLDER = league_cfg["folder"]
MODEL_PATH  = league_cfg["model"]

print(f"\n已选择：{league_cfg['name']}")
print("正在加载历史数据和模型...")
df_hist = load_multiple_seasons(DATA_FOLDER)
df_hist = df_hist.dropna(subset=["B365H", "B365D", "B365A"])
df_hist["Date"] = pd.to_datetime(df_hist["Date"], dayfirst=True, errors="coerce")
df_hist = df_hist.sort_values("Date").reset_index(drop=True)
model = joblib.load(MODEL_PATH)
print("加载完成！\n")

# =========================
# 按需加载联赛（供 web 调用）
# =========================
def get_strong_avg_shots(df, team, today):
    rows = df[
        ((df["HomeTeam"] == team) & (df["B365H"] < df["B365A"])) |
        ((df["AwayTeam"] == team) & (df["B365A"] < df["B365H"]))
    ].copy()
    rows = rows[rows["Date"] < today].sort_values("Date").tail(10)
    shots = []
    for _, r in rows.iterrows():
        shots.append(r["HS"] if r["HomeTeam"] == team else r["AS"])
    return np.mean(shots) if len(shots) >= 3 else np.nan

def get_avg_corners(df, team, is_home, today):
    """取某队作为主/客场时近10场平均角球数"""
    col = "HC" if is_home else "AC"
    team_col = "HomeTeam" if is_home else "AwayTeam"
    rows = df[(df[team_col] == team) & (df["Date"] < today)].sort_values("Date").tail(10)
    return np.mean(rows[col].dropna()) if len(rows) >= 3 else np.nan

def get_weak_scores(df, weak_team, today):
    past = df[
        (df["Date"] < today) &
        (
            ((df["HomeTeam"] == weak_team) & (df["B365A"] < df["B365H"])) |
            ((df["AwayTeam"] == weak_team) & (df["B365H"] < df["B365A"]))
        )
    ].copy()

    print(f"  弱队历史打强队场次: {len(past)}")

    if len(past) < 5:
        return np.nan, np.nan

    shots_for, shots_against = [], []
    for _, m in past.iterrows():
        if m["HomeTeam"] == weak_team:
            shots_for.append(m["HS"])
            shots_against.append(m["AS"])
        else:
            shots_for.append(m["AS"])
            shots_against.append(m["HS"])
    return np.mean(shots_for), -np.mean(shots_against)

def predict(home_team, away_team, b365h, b365d, b365a, b365ch, b365cd, b365ca,
            psh, psd, psa, today，df_hist=None, model=None):

    strong_is_home = b365h < b365a

    strong_win_odds = b365h if strong_is_home else b365a
    weak_win_odds   = b365a if strong_is_home else b365h
    draw_odds       = b365d
    strong_close    = b365ch if strong_is_home else b365ca
    weak_close      = b365ca if strong_is_home else b365ch

    strong_team = home_team if strong_is_home else away_team
    weak_team   = away_team if strong_is_home else home_team

    print(f"\n强队: {strong_team} (赔率 {strong_win_odds})")
    print(f"弱队: {weak_team} (赔率 {weak_win_odds})")

    if strong_win_odds > 1.8 or weak_win_odds < 4:
        print(f"⚠️  不符合筛选条件（强队赔率需≤1.8，弱队赔率需≥4），结果仅供参考")

    # B365 去水概率
    inv_h, inv_d, inv_a = 1/b365h, 1/b365d, 1/b365a
    total    = inv_h + inv_d + inv_a
    p_strong = (inv_h if strong_is_home else inv_a) / total
    implied_strong = p_strong
    implied_draw   = inv_d / total
    implied_weak   = (inv_a if strong_is_home else inv_h) / total

    # Pinnacle 去水概率 & 分歧
    inv_ps_h, inv_ps_d, inv_ps_a = 1/psh, 1/psd, 1/psa
    ps_total    = inv_ps_h + inv_ps_d + inv_ps_a
    ps_p_strong = (inv_ps_h if strong_is_home else inv_ps_a) / ps_total
    odds_market_diff = p_strong - ps_p_strong

    # 赔率变化
    strong_odds_move = strong_close - strong_win_odds
    weak_odds_move   = weak_close   - weak_win_odds
    strong_move_pct  = strong_odds_move / strong_win_odds
    weak_move_pct    = weak_odds_move   / weak_win_odds

    # 历史统计
    strong_avg_shots = get_strong_avg_shots(df_hist, strong_team, today)
    print(f"  强队近10场平均射门: {round(strong_avg_shots, 2) if not np.isnan(strong_avg_shots) else '数据不足'}")

    weak_attack_score, weak_defense_score = get_weak_scores(df_hist, weak_team, today)
    if not np.isnan(weak_attack_score):
        print(f"  弱队抗强进攻评分: {round(weak_attack_score, 2)}")
        print(f"  弱队抗强防守评分: {round(weak_defense_score, 2)}")
    else:
        print("  ⚠️  弱队历史打强队数据不足5场，相关特征为NaN，结果仅供参考")

    # 角球
    strong_avg_corners = get_avg_corners(df_hist, strong_team, bool(strong_is_home), today)
    weak_avg_corners   = get_avg_corners(df_hist, weak_team, not strong_is_home, today)
    print(f"  强队近10场平均角球: {round(strong_avg_corners, 2) if not np.isnan(strong_avg_corners) else '数据不足'}")
    print(f"  弱队近10场平均角球: {round(weak_avg_corners, 2) if not np.isnan(weak_avg_corners) else '数据不足'}")
    print(f"  B365 vs Pinnacle 强队概率分歧: {round(odds_market_diff, 4):+.4f}")

    features = {
        "strong_win_odds":    strong_win_odds,
        "draw_odds":          draw_odds,
        "weak_win_odds":      weak_win_odds,
        "odds_gap":           weak_win_odds - strong_win_odds,
        "odds_ratio":         weak_win_odds / strong_win_odds,
        "p_strong":           p_strong,
        "strong_is_home":     int(strong_is_home),
        "strong_odds_move":   strong_odds_move,
        "weak_odds_move":     weak_odds_move,
        "strong_move_pct":    strong_move_pct,
        "weak_move_pct":      weak_move_pct,
        "strong_avg_shots":   strong_avg_shots,
        "weak_attack_score":  weak_attack_score,
        "weak_defense_score": weak_defense_score,
        "ps_p_strong":        ps_p_strong,
        "odds_market_diff":   odds_market_diff,
        "strong_avg_corners": strong_avg_corners,
        "weak_avg_corners":   weak_avg_corners,
    }

    X_pred = pd.DataFrame([features])[FEATURES]
    prob        = model.predict_proba(X_pred)[0][1]   # 冷门概率（强队不胜）
    prob_strong = 1 - prob                             # 强队胜概率

    # 去水隐含概率
    inv_h, inv_d, inv_a = 1/b365h, 1/b365d, 1/b365a
    total          = inv_h + inv_d + inv_a
    implied_strong = (inv_h if strong_is_home else inv_a) / total
    implied_draw   = inv_d / total
    implied_weak   = (inv_a if strong_is_home else inv_h) / total

    # 强队EV
    ev_strong = prob_strong * strong_win_odds - 1

    # 冷门分歧：模型冷门概率 vs 市场隐含冷门概率
    upset_diff = prob - (implied_draw + implied_weak)

    print("\n" + "="*45)
    print(f"🏟️  {home_team} vs {away_team}")
    print(f"\n── 概率对比 ──")
    print(f"  {'':<12}  {'模型':>8}  {'市场隐含':>8}")
    print(f"  {'强队赢':<12}  {prob_strong*100:>7.1f}%  {implied_strong*100:>7.1f}%")
    print(f"  {'平局+冷门':<12}  {prob*100:>7.1f}%  {(implied_draw+implied_weak)*100:>7.1f}%")
    print(f"  {'  其中冷门':<12}  {'':>8}  {implied_weak*100:>7.1f}%")
    print(f"  模型 vs 市场「强队不胜」分歧: {upset_diff*100:>+.1f}%")

    print(f"\n── 强队EV ──")
    print(f"  下注强队赢  赔率 {strong_win_odds:<5}  EV: {ev_strong:>+.3f}  {'✅ 正期望' if ev_strong > 0 else '❌ 负期望'}")

    # 冷门分歧阈值：模型比市场高出超过8%认为有翻车风险
    UPSET_THRESHOLD = 0.08
    # 把建议部分替换成以下代码

    print(f"\n── 建议 ──")

    if ev_strong > 0 and upset_diff >= UPSET_THRESHOLD:
        # 两个信号矛盾：EV正但冷门风险高
        print(f"  ⚠️  信号矛盾：强队EV正期望，但冷门风险显著高于市场(+{upset_diff * 100:.1f}%)")
        print(f"  👉 本场存在不确定性，建议跳过")
        _rec = "信号矛盾，建议跳过"

    elif upset_diff >= UPSET_THRESHOLD and ev_strong <= 0:
        # 冷门风险高 且 EV为负，两个信号一致指向弱队
        print(f"  ⚠️  模型认为冷门概率显著高于市场(+{upset_diff * 100:.1f}%)")
        print(f"  👉 强队有翻车风险，建议考虑弱队({weak_team})让球盘")
        _rec = f"冷门风险高，考虑弱队({weak_team})"

    elif ev_strong > 0 and upset_diff < UPSET_THRESHOLD:
        # 冷门风险低 且 EV正，两个信号一致指向强队
        print(f"  冷门风险低，强队EV正期望")
        print(f"  👉 可考虑强队({strong_team})赢")
        print(f"     每投 100 元，长期期望盈利 {round(ev_strong * 100, 1)} 元")
        _rec = f"强队EV正，考虑强队({strong_team})"

    else:
        # 冷门风险低 但 EV为负
        print(f"  冷门风险低，但强队EV为负")
        print(f"  👉 本场建议跳过")
        _rec = "建议跳过"



        # ── 自动保存预测记录（供反馈学习使用）──
    _features_dict = {
        "strong_win_odds": strong_win_odds,
        "draw_odds": draw_odds,
        "weak_win_odds": weak_win_odds,
        "odds_gap": weak_win_odds - strong_win_odds,
        "odds_ratio": weak_win_odds / strong_win_odds,
        "p_strong": p_strong,
        "strong_is_home": int(strong_is_home),
        "strong_odds_move": strong_odds_move,
        "weak_odds_move": weak_odds_move,
        "strong_move_pct": strong_move_pct,
        "weak_move_pct": weak_move_pct,
        "strong_avg_shots": strong_avg_shots,
        "weak_attack_score": weak_attack_score,
        "weak_defense_score": weak_defense_score,
        "ps_p_strong": ps_p_strong,
        "odds_market_diff": odds_market_diff,
        "strong_avg_corners": strong_avg_corners,
        "weak_avg_corners": weak_avg_corners,
    }

    # 建议字符串
    if upset_diff >= UPSET_THRESHOLD:
        _rec = f"冷门风险高，考虑弱队({weak_team})"
    elif ev_strong > 0:
        _rec = f"强队EV正，考虑强队({strong_team})"
    else:
        _rec = "建议跳过"


    print("="*45)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "strong_team": strong_team,
        "weak_team": weak_team,
        "prob_strong": prob_strong,
        "prob_upset": prob,
        "ev_strong": ev_strong,
        "recommendation": _rec
    }


# =========================
# 交互输入
# =========================
def input_float(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  请输入有效数字！")

def input_date(prompt):
    while True:
        s = input(prompt).strip()
        try:
            return pd.Timestamp(s)
        except Exception:
            print("  日期格式错误，请用 YYYY-MM-DD 格式！")

def run():

    while True:
        print("\n" + "─"*45)
        print("       冷门预测系统")
        print("─"*45)

        home_team = input("主队名称: ").strip()
        away_team = input("客队名称: ").strip()

        print("\n── 开盘赔率 ──")
        b365h = input_float("主队赔率 (B365H): ")
        b365d = input_float("平局赔率 (B365D): ")
        b365a = input_float("客队赔率 (B365A): ")

        print("\n── 终盘赔率 ──")
        b365ch = input_float("主队终盘赔率 (B365CH): ")
        b365cd = input_float("平局终盘赔率 (B365CD): ")
        b365ca = input_float("客队终盘赔率 (B365CA): ")

        print("\n── Pinnacle 赔率 ──")
        psh = input_float("主队赔率 (PSH): ")
        psd = input_float("平局赔率 (PSD): ")
        psa = input_float("客队赔率 (PSA): ")

        today = input_date("\n比赛日期 (YYYY-MM-DD): ")

        predict(
            home_team, away_team,
            b365h, b365d, b365a,
            b365ch, b365cd, b365ca,
            psh, psd, psa,
            today
        )

        again = input("\n继续预测另一场？(y/n): ").strip().lower()
        if again != "y":
            print("退出预测系统。")
            break


if __name__ == "__main__":

    run()





