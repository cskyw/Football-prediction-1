import pandas as pd
import os


def load_multiple_seasons(folder_path):
    """
    读取文件夹下所有CSV文件
    按行合并
    不添加Season列
    """

    all_dfs = []

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):

            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            # 日期统一格式
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

            all_dfs.append(df)

            print(f"已加载: {file} 行数: {len(df)}")

    if not all_dfs:
        raise ValueError("文件夹中没有CSV文件")

    df_all = pd.concat(all_dfs, ignore_index=True)

    # 按时间排序（重要）
    df_all = df_all.sort_values("Date").reset_index(drop=True)

    print("合并完成，总行数:", len(df_all))

    return df_all