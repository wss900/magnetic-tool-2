"""
谐波斜率曲率分析（单文件版，接收上传文件内容）
功能：
- 从上传的 .txt 文件中提取电流值
- 二次拟合求 B2 曲率，线性拟合求斜率
- 计算比值并生成曲线图
- 返回 Excel 结果文件和图片
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
import io
import zipfile
import os

def process_files(uploaded_files, x_min, x_max):
    """处理上传的文件列表，返回 (结果DataFrame, 图片二进制数据)"""
    results = []
    plot_data = {'mA': [], 'ratio': []}

    for uploaded in uploaded_files:
        filename = uploaded.name
        # 提取电流值
        try:
            match = re.search(r'(\d+\.?\d*)mA', filename)
            current_value = float(match.group(1)) if match else float('nan')
        except:
            current_value = float('nan')

        # 读取文件内容
        try:
            content = uploaded.read().decode('utf-8')
            data = pd.read_csv(io.StringIO(content), sep=r'\s+', header=None)
        except:
            continue

        if data.shape[1] < 3:
            continue

        col1 = pd.to_numeric(data.iloc[:, 0], errors='coerce')
        col2 = pd.to_numeric(data.iloc[:, 1], errors='coerce')
        col3 = pd.to_numeric(data.iloc[:, 2], errors='coerce')

        mask = (col1 >= x_min) & (col1 <= x_max)
        x = col1[mask].values
        y2 = col2[mask].values
        y3 = col3[mask].values

        # 二次拟合
        def quad(x, a, b, c):
            return a*x**2 + b*x + c
        try:
            popt_q, _ = curve_fit(quad, x, y2)
            B2 = popt_q[0]
        except:
            B2 = float('nan')

        # 线性拟合
        def lin(x, m, c):
            return m*x + c
        try:
            popt_l, _ = curve_fit(lin, x, y3)
            slope = popt_l[0]
        except:
            slope = float('nan')

        if not np.isnan(B2) and not np.isnan(slope) and B2 != 0:
            ratio = -slope / B2
        else:
            ratio = float('nan')

        results.append({
            'filename': filename,
            'current_mA': current_value,
            'B2_curvature': B2,
            'linear_slope': slope,
            'ratio': ratio
        })
        if not np.isnan(current_value) and not np.isnan(ratio):
            plot_data['mA'].append(current_value)
            plot_data['ratio'].append(ratio)

    df = pd.DataFrame(results)

    # 生成图片
    fig, ax = plt.subplots(figsize=(8, 6))
    if plot_data['mA']:
        mA = np.array(plot_data['mA'])
        ratio = np.array(plot_data['ratio'])
        sort_idx = np.argsort(mA)
        mA_s = mA[sort_idx]
        ratio_s = ratio[sort_idx]

        pos_mask = ratio_s > 0
        neg_mask = ratio_s < 0

        # 正比值点及过原点拟合
        if np.any(pos_mask):
            xp = mA_s[pos_mask]
            yp = ratio_s[pos_mask]
            ax.scatter(xp, yp, c='red', label='Positive')
            if len(xp) >= 2:
                kp = np.sum(xp*yp) / np.sum(xp*xp)
                x_fit = np.linspace(0, max(xp), 50)
                ax.plot(x_fit, kp*x_fit, 'r--', label=f'k={kp:.4f}')

        # 负比值
        if np.any(neg_mask):
            xn = mA_s[neg_mask]
            yn = ratio_s[neg_mask]
            ax.scatter(xn, yn, c='blue', label='Negative')
            if len(xn) >= 2:
                kn = np.sum(xn*yn) / np.sum(xn*xn)
                x_fit = np.linspace(0, max(xn), 50)
                ax.plot(x_fit, kn*x_fit, 'b--', label=f'k={kn:.4f}')

        ax.set_xlabel('Current (mA)')
        ax.set_ylabel('-slope / B2 curvature')
        ax.axhline(0, color='grey')
        ax.axvline(0, color='grey')
        ax.legend()
        ax.set_title('Ratio vs Current')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)

    return df, buf


if __name__ == "__main__":
    # 被 Streamlit 调用时不会执行这里，留着给本地测试用
    pass
