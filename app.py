from flask import Flask, render_template, request
from EV_prediction_3 import predict
from dataloader import load_multiple_seasons
import joblib

app = Flask(__name__)

# ===== 启动时加载数据（与命令行版一致，联赛在 EV_prediction_3.py 顶部选定）=====
from EV_prediction_3 import DATA_FOLDER, MODEL_PATH, df_hist, model, league_cfg

print("服务器启动完成")


@app.route("/", methods=["GET", "POST"])
def index():

    # ✅ 修复：统一用 result 变量，GET 时为 None，不会报错
    result = None
    output = None

    if request.method == "POST":

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

        # ✅ 与命令行版完全一致的 predict 调用
        result = predict(
            home_team, away_team,
            b365h, b365d, b365a,
            b365ch, b365cd, b365ca,
            psh, psd, psa,
            date
        )

        # ✅ 格式化显示字符串，与命令行版输出逻辑一致
        output = {
            "match":        f"{result['home_team']} vs {result['away_team']}",
            "strong_team":  result['strong_team'],
            "weak_team":    result['weak_team'],
            "prob_strong":  f"{result['prob_strong'] * 100:.1f}%",
            "prob_upset":   f"{result['prob_upset'] * 100:.1f}%",
            "ev_strong":    f"{result['ev_strong']:.3f}",
            "ev_positive":  result['ev_strong'] > 0,
            "recommendation": result['recommendation'],
        }

    return render_template("index.html", output=output)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)