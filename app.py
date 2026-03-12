from flask import Flask, render_template, request
from EV_prediction_3 import predict, load_league, LEAGUES
import os

app = Flask(__name__)

# 缓存字典：避免每次请求都重新加载
_cache = {}  # { "1": (df_hist, model, cfg), ... }

def get_league(choice):
    if choice not in _cache:
        print(f"首次加载联赛 {LEAGUES[choice]['name']}，请稍候...")
        _cache[choice] = load_league(choice)
        print(f"{LEAGUES[choice]['name']} 加载完成")
    return _cache[choice]

print("服务器启动完成")

@app.route("/", methods=["GET", "POST"])
def index():
    output = None
    selected_league = "2"  # 默认英冠

    if request.method == "POST":
        selected_league = request.form.get("league", "2")

        home_team = request.form["home_team"]
        away_team = request.form["away_team"]
        date      = request.form["date"]
        b365h  = float(request.form["b365h"])
        b365d  = float(request.form["b365d"])
        b365a  = float(request.form["b365a"])
        b365ch = float(request.form["b365ch"])
        b365cd = float(request.form["b365cd"])
        b365ca = float(request.form["b365ca"])
        psh = float(request.form["psh"])
        psd = float(request.form["psd"])
        psa = float(request.form["psa"])

        df_hist, model, cfg = get_league(selected_league)

        result = predict(
            home_team, away_team,
            b365h, b365d, b365a,
            b365ch, b365cd, b365ca,
            psh, psd, psa,
            date,
            df_hist=df_hist,
            model=model,
        )

        output = {
            "match":        f"{result['home_team']} vs {result['away_team']}",
            "league_name":  cfg["name"],
            "strong_team":  result["strong_team"],
            "weak_team":    result["weak_team"],
            "prob_strong":  f"{result['prob_strong'] * 100:.1f}%",
            "prob_upset":   f"{result['prob_upset'] * 100:.1f}%",
            "ev_strong":    f"{result['ev_strong']:.3f}",
            "ev_positive":  result["ev_strong"] > 0,
            "recommendation": result["recommendation"],
        }

    return render_template("index.html",
                           output=output,
                           leagues=LEAGUES,
                           selected_league=selected_league)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
