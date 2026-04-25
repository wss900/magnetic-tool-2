import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re
import os
from scipy.optimize import curve_fit
from collections import OrderedDict
import zipfile

st.set_page_config(page_title="PPMS 数据拟合", page_icon="🧪")
st.title("🧪 PPMS 角度扫描数据拟合")

st.markdown("""
**功能**：上传 PPMS 角度扫描 TXT 文件（支持多文件），自动识别 Mag Field、Angle、Lock-in X 列，
按 Mag 分组，对每组 Angle vs Lock-in X 数据用七参数三角函数拟合，输出处理后数据、拟合参数和拟合曲线图。
- 输出：处理后数据 Excel、拟合参数 Excel、合并拟合曲线图 PNG
- 打包为 ZIP 下载
""")

uploaded_files = st.file_uploader("📂 选择或拖拽 .txt 文件", type="txt", accept_multiple_files=True)

col1, col2, col3 = st.columns(3)
with col1:
    sort_angle = st.selectbox("Angle 排序", ["True", "False"], index=0)
with col2:
    enable_fitting = st.selectbox("开启拟合", ["True", "False"], index=0)
with col3:
    generate_plot = st.selectbox("生成合并图", ["True", "False"], index=0)


# 七参数拟合函数
def fitting_func(x, A, B, C, D, E, F, G):
    theta = (x + A) * np.pi / 180
    term1 = B * np.cos(theta)
    term2 = C * np.cos(2 * theta) * np.cos(theta)
    term3 = D * np.cos(2 * theta)
    term4 = E * np.sin(theta)
    term5 = F * np.sin(2 * theta)
    return term1 + term2 + term3 + term4 + term5 + G


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def process_single_file(content, filename, sort_angle=True, enable_fitting=True, generate_plot=True):
    lines = content.strip().split('\n')
    # 寻找表头
    header_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if 'Mag Field' in line and 'Angle' in line and 'Lock-in' in line:
            header_idx = i
            break
    if header_idx == -1:
        raise ValueError("未找到符合要求的表头")

    headers = re.split(r'\t|\s{2,}', lines[header_idx].strip())
    headers = [h.strip() for h in headers if h.strip()]

    mag_col = next(i for i, h in enumerate(headers) if 'Mag Field' in h)
    angle_col = next(i for i, h in enumerate(headers) if 'Angle' in h)
    lockin_x_cols = [i for i, h in enumerate(headers) if 'Lock-in' in h and ('x' in h or 'X' in h)]
    lockin_col = lockin_x_cols[1] if len(lockin_x_cols) >= 2 else lockin_x_cols[0]

    all_data = []
    for line in lines[header_idx + 1:]:
        cols = re.split(r'\t|\s{2,}', line.strip())
        cols = [c.strip() for c in cols if c.strip()]
        if len(cols) <= max(mag_col, angle_col, lockin_col):
            continue
        try:
            mag = float(cols[mag_col])
            angle = float(cols[angle_col])
            lockin = float(cols[lockin_col])
            all_data.append((mag, angle, lockin))
        except:
            continue

    if not all_data:
        raise ValueError("未提取到有效数据")

    # Mag 分组
    all_mags = sorted([d[0] for d in all_data])
    mag_groups = []
    current_group = [all_mags[0]]
    for mag in all_mags[1:]:
        if abs(mag - current_group[-1]) <= 2:
            current_group.append(mag)
        else:
            mag_groups.append(current_group)
            current_group = [mag]
    if current_group:
        mag_groups.append(current_group)

    processed_groups = OrderedDict()
    for g in mag_groups:
        mag_rep = round(np.mean(g), 0)
        group_data = [d for d in all_data if d[0] in g]
        angle_last = OrderedDict()
        for mag, angle, lockin in group_data:
            angle_last[angle] = lockin
        angle_list = list(angle_last.items())
        if sort_angle:
            angle_list.sort(key=lambda x: x[0])
        processed_groups[mag_rep] = angle_list

    # 构建输出表格
    max_rows = max(len(g) for g in processed_groups.values())
    output_rows = []
    row1 = [""]
    for mag_rep in processed_groups.keys():
        row1.extend([mag_rep, mag_rep])
    output_rows.append(row1)
    row2 = ["Angle"]
    for _ in processed_groups.keys():
        row2.extend(["Angle", "Lock-inx"])
    output_rows.append(row2)
    for row_idx in range(max_rows):
        current_row = [""]
        for mag_rep in processed_groups.keys():
            group_data = processed_groups[mag_rep]
            if row_idx < len(group_data):
                angle, lockin = group_data[row_idx]
                current_row.extend([angle, lockin])
            else:
                current_row.extend([np.nan, np.nan])
        output_rows.append(current_row)

    df_processed = pd.DataFrame(output_rows)

    # 拟合
    fit_results = []
    if enable_fitting:
        initial_guess = [-90, 0, 0, 0, 0, 0, 0]
        for mag_rep, angle_list in processed_groups.items():
            x = np.array([a[0] for a in angle_list if not pd.isna(a[0]) and not pd.isna(a[1])])
            y = np.array([a[1] for a in angle_list if not pd.isna(a[0]) and not pd.isna(a[1])])
            y = y * 1e6  # 转换为 μV
            if len(x) < 7:
                fit_results.append({
                    'Mag': mag_rep,
                    'A': np.nan, 'B': np.nan, 'C': np.nan, 'D': np.nan,
                    'E': np.nan, 'F': np.nan, 'G': np.nan, 'R²': np.nan
                })
                continue
            try:
                popt, _ = curve_fit(fitting_func, x, y, p0=initial_guess, maxfev=10000)
                A, B, C, D, E, F, G = popt
                y_pred = fitting_func(x, *popt)
                r2 = calc_r2(y, y_pred)
                fit_results.append({
                    'Mag': mag_rep,
                    'A': A, 'B': B, 'C': C, 'D': D,
                    'E': E, 'F': F, 'G': G, 'R²': r2
                })
            except Exception as e:
                fit_results.append({
                    'Mag': mag_rep,
                    'A': np.nan, 'B': np.nan, 'C': np.nan, 'D': np.nan,
                    'E': np.nan, 'F': np.nan, 'G': np.nan, 'R²': np.nan
                })
        df_fit = pd.DataFrame(fit_results)
    else:
        df_fit = pd.DataFrame()

    # 绘图
    img_buf = None
    if enable_fitting and generate_plot and fit_results:
        num_groups = len(processed_groups)
        cols_plot = min(3, num_groups)
        rows_plot = (num_groups + cols_plot - 1) // cols_plot
        fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(cols_plot * 8, rows_plot * 6))
        if num_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        colors = plt.cm.viridis(np.linspace(0, 1, num_groups))
        for idx, (mag_rep, angle_list) in enumerate(processed_groups.items()):
            x = np.array([a[0] for a in angle_list if not pd.isna(a[0]) and not pd.isna(a[1])])
            y = np.array([a[1] for a in angle_list if not pd.isna(a[0]) and not pd.isna(a[1])]) * 1e6
            ax = axes[idx]
            ax.scatter(x, y, color=colors[idx], label=f'数据 (Mag={mag_rep}Oe)', s=50)
            fit_row = next((r for r in fit_results if r['Mag'] == mag_rep), None)
            if fit_row and not np.isnan(fit_row['R²']):
                popt = [fit_row[p] for p in ['A','B','C','D','E','F','G']]
                x_fit = np.linspace(min(x), max(x), 500)
                y_fit = fitting_func(x_fit, *popt)
                ax.plot(x_fit, y_fit, 'r-', label=f"拟合 (R²={fit_row['R²']:.4f})")
            ax.set_title(f'Mag = {mag_rep} Oe')
            ax.set_xlabel('Angle (°)')
            ax.set_ylabel('Lock-in X (μV)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        img_buf = buf
        plt.close(fig)

    return df_processed, df_fit, img_buf


if st.button("▶️ 开始处理", use_container_width=True):
    if not uploaded_files:
        st.error("请至少上传一个 .txt 文件")
    else:
        all_processed = {}
        with st.spinner("正在处理..."):
            for uploaded in uploaded_files:
                filename = uploaded.name
                try:
                    content = uploaded.read().decode('utf-8', errors='ignore')
                    sort_flag = sort_angle == "True"
                    fit_flag = enable_fitting == "True"
                    plot_flag = generate_plot == "True"
                    df_proc, df_fit, img = process_single_file(
                        content, filename, sort_flag, fit_flag, plot_flag
                    )
                    all_processed[filename] = (df_proc, df_fit, img)
                except Exception as e:
                    st.warning(f"处理 {filename} 失败: {e}")
        if all_processed:
            # 打包所有结果为一个 ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fname, (df_p, df_f, img_buf) in all_processed.items():
                    base = os.path.splitext(fname)[0]
                    # 处理后数据 Excel
                    excel1 = io.BytesIO()
                    with pd.ExcelWriter(excel1, engine='openpyxl') as writer:
                        df_p.to_excel(writer, index=False, header=False)
                    excel1.seek(0)
                    zf.writestr(f"{base}_processed.xlsx", excel1.getvalue())
                    # 拟合参数 Excel
                    if not df_f.empty:
                        excel2 = io.BytesIO()
                        with pd.ExcelWriter(excel2, engine='openpyxl') as writer:
                            df_f.to_excel(writer, index=False)
                        excel2.seek(0)
                        zf.writestr(f"{base}_fitted.xlsx", excel2.getvalue())
                    # 拟合图
                    if img_buf is not None:
                        zf.writestr(f"{base}_fit_plot.png", img_buf.getvalue())
            zip_buffer.seek(0)
            st.success("处理完成！")
            st.download_button(
                label="⬇️ 下载所有结果 (ZIP)",
                data=zip_buffer,
                file_name="ppms_results.zip",
                mime="application/zip"
            )
        else:
            st.warning("没有文件被成功处理")
