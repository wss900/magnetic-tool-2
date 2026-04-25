import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import re
import zipfile
from scipy.optimize import curve_fit
from collections import OrderedDict
import glob

# ---------------- 页面配置 ----------------
st.set_page_config(page_title="磁性测量数据处理平台", page_icon="🧲", layout="wide")

# ---------------- 自定义样式 ----------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .designer {
        font-size: 0.9rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: 700;
        padding: 0.6rem 2rem;
        border-radius: 0.5rem;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #145a8c;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- 标题与设计人 ----------------
st.markdown('<div class="main-header">🧲 磁性测量数据处理平台</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">选择任务，上传文件，填写参数，一键运行</div>', unsafe_allow_html=True)
st.markdown('<div class="designer">设计人：王赛赛</div>', unsafe_allow_html=True)

# ---------------- 处理函数定义 ----------------

# 1. 谐波斜率曲率分析
def harmonic_slope_process(uploaded_files, x_min, x_max):
    results = []
    plot_data = {'mA': [], 'ratio': []}
    for uploaded in uploaded_files:
        filename = uploaded.name
        try:
            match = re.search(r'(\d+\.?\d*)mA', filename)
            current = float(match.group(1)) if match else float('nan')
        except:
            current = float('nan')
        try:
            content = uploaded.read().decode('utf-8')
            data = pd.read_csv(io.StringIO(content), sep=r'\s+', header=None)
        except:
            continue
        if data.shape[1] < 3:
            continue
        col1 = pd.to_numeric(data.iloc[:,0], errors='coerce')
        col2 = pd.to_numeric(data.iloc[:,1], errors='coerce')
        col3 = pd.to_numeric(data.iloc[:,2], errors='coerce')
        mask = (col1 >= x_min) & (col1 <= x_max)
        x = col1[mask].values
        y2 = col2[mask].values
        y3 = col3[mask].values

        def quad(x, a, b, c): return a*x**2 + b*x + c
        def lin(x, m, c): return m*x + c
        try:
            popt_q, _ = curve_fit(quad, x, y2)
            B2 = popt_q[0]
        except:
            B2 = float('nan')
        try:
            popt_l, _ = curve_fit(lin, x, y3)
            slope = popt_l[0]
        except:
            slope = float('nan')
        ratio = (-slope / B2) if (not np.isnan(B2) and not np.isnan(slope) and B2 != 0) else float('nan')
        results.append({
            'filename': filename,
            'current_mA': current,
            'B2_curvature': B2,
            'linear_slope': slope,
            'ratio': ratio
        })
        if not np.isnan(current) and not np.isnan(ratio):
            plot_data['mA'].append(current)
            plot_data['ratio'].append(ratio)

    df = pd.DataFrame(results)
    # 绘图
    fig, ax = plt.subplots(figsize=(8,6))
    if plot_data['mA']:
        mA = np.array(plot_data['mA'])
        ratio = np.array(plot_data['ratio'])
        sort_idx = np.argsort(mA)
        mA_s = mA[sort_idx]
        ratio_s = ratio[sort_idx]
        pos = ratio_s > 0
        neg = ratio_s < 0
        if np.any(pos):
            xp = mA_s[pos]; yp = ratio_s[pos]
            ax.scatter(xp, yp, c='red', label='Positive')
            if len(xp) >= 2:
                kp = np.sum(xp*yp) / np.sum(xp*xp)
                ax.plot(np.linspace(0, max(xp), 50), kp*np.linspace(0, max(xp), 50), 'r--', label=f'k={kp:.4f}')
        if np.any(neg):
            xn = mA_s[neg]; yn = ratio_s[neg]
            ax.scatter(xn, yn, c='blue', label='Negative')
            if len(xn) >= 2:
                kn = np.sum(xn*yn) / np.sum(xn*xn)
                ax.plot(np.linspace(0, max(xn), 50), kn*np.linspace(0, max(xn), 50), 'b--', label=f'k={kn:.4f}')
        ax.axhline(0, color='grey'); ax.axvline(0, color='grey')
        ax.set_xlabel('Current (mA)'); ax.set_ylabel('-slope / B2 curvature')
        ax.legend(); ax.set_title('Ratio vs Current')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return df, buf

# 2. 单侧谐波跳点去除
def harmonic_remove_spikes(uploaded_files, z_thresh, diff_thresh, use_first, first_diff_thresh,
                           detect_sign, sign_thresh, suffix):
    cleaned_data = {}
    for uploaded in uploaded_files:
        filename = uploaded.name
        try:
            content = uploaded.read().decode('utf-8')
            df = pd.read_csv(io.StringIO(content), sep='\t', header=None)
        except:
            continue
        if df.shape[1] < 3:
            continue
        try:
            v1 = pd.to_numeric(df.iloc[:,1], errors='coerce').values
            v2 = pd.to_numeric(df.iloc[:,2], errors='coerce').values
        except:
            continue
        n = len(v2)
        mask = np.ones(n, dtype=bool)

        if z_thresh is not None:
            mean_v2 = np.mean(v2); std_v2 = np.std(v2)
            if std_v2 > 0:
                z = np.abs((v2 - mean_v2) / std_v2)
                mask &= (z < z_thresh)

        if diff_thresh is not None:
            dfwd = np.abs(np.diff(v2, prepend=v2[0]))
            dbwd = np.abs(np.diff(v2, append=v2[-1]))
            dmax = np.maximum(dfwd, dbwd)
            mask &= (dmax < diff_thresh)

        if detect_sign:
            sign_changes = np.zeros(n, dtype=bool)
            for i in range(1, n):
                if v2[i]*v2[i-1] < 0 and abs(v2[i]-v2[i-1]) > sign_thresh:
                    sign_changes[i] = True; sign_changes[i-1] = True
            mask &= ~sign_changes

        if use_first:
            dv1 = np.abs(np.diff(v1, prepend=v1[0]))
            mask &= (dv1 < first_diff_thresh)

        df_clean = df[mask].reset_index(drop=True)
        out_buf = io.BytesIO()
        df_clean.to_csv(out_buf, sep='\t', index=False, header=False)
        out_buf.seek(0)
        base, ext = os.path.splitext(filename)
        cleaned_data[f"{base}{suffix}{ext}"] = out_buf

    if not cleaned_data:
        return None
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, buf in cleaned_data.items():
            zf.writestr(name, buf.getvalue())
    zip_buf.seek(0)
    return zip_buf

# 3. 谐波提取上升/下降段
def extract_segments(uploaded_file, range_min, range_max, tolerance):
    content = uploaded_file.read().decode('utf-8')
    lines = content.strip().split('\n')
    asc = []
    found_min = False; found_max = False
    for line in lines:
        parts = line.split()
        if not parts: continue
        try:
            val = float(parts[0])
        except: continue
        if not found_min:
            if abs(val - range_min) < tolerance:
                found_min = True; asc.append(line)
                continue
        else:
            asc.append(line)
            if abs(val - range_max) < tolerance:
                found_max = True; break
    if not found_min or not found_max:
        return None, None

    try: after_idx = lines.index(asc[-1]) + 1
    except: after_idx = len(lines)
    desc = []
    found_max = False; found_min = False
    for i in range(after_idx, len(lines)):
        line = lines[i]
        parts = line.split()
        if not parts: continue
        try:
            val = float(parts[0])
        except: continue
        if not found_max:
            if abs(val - range_max) < tolerance:
                found_max = True; desc.append(line)
                continue
        else:
            desc.append(line)
            if abs(val - range_min) < tolerance:
                found_min = True; break
    if not found_max or not found_min:
        return None, None
    return '\n'.join(asc), '\n'.join(desc)

# 4. Excel 批量合并（跳过行）
def excel_merge(uploaded_files, skip_rows):
    data_list = []; file_names = []
    for uploaded in uploaded_files:
        try:
            df = pd.read_excel(uploaded)
        except: continue
        if df.empty or df.shape[1] < 4: continue
        x = pd.to_numeric(df.iloc[:,2], errors='coerce')
        y = pd.to_numeric(df.iloc[:,3], errors='coerce')
        base = os.path.splitext(uploaded.name)[0]
        temp = pd.DataFrame({f"{base}_X": np.abs(x), f"{base}_Y": np.abs(y)}).dropna()
        if len(temp) > skip_rows:
            temp = temp.iloc[skip_rows:].reset_index(drop=True)
            data_list.append(temp)
            file_names.append(base)
    if not data_list: return None
    all_data = pd.concat(data_list, axis=1)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        all_data.to_excel(writer, index=False)
    out.seek(0)
    return out

# 5. PPMS 数据拟合（简化版，支持多文件上传）
def ppms_fit(uploaded_files, sort_angle, enable_fitting, generate_plot):
    def fit_func(x, A, B, C, D, E, F, G):
        theta = (x + A) * np.pi / 180
        return (B * np.cos(theta) +
                C * np.cos(2*theta) * np.cos(theta) +
                D * np.cos(2*theta) +
                E * np.sin(theta) +
                F * np.sin(2*theta) + G)

    def calc_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    all_results = []
    figs = []
    for uploaded in uploaded_files:
        filename = uploaded.name
        try:
            content = uploaded.read().decode('utf-8', errors='ignore')
            lines = [line for line in content.split('\n') if line.strip()]
        except:
            continue

        # 寻找最后一个有效表头
        header_idx = -1
        for i in range(len(lines)-1, -1, -1):
            if 'Mag Field' in lines[i] and 'Angle' in lines[i] and 'Lock-in' in lines[i]:
                header_idx = i
                break
        if header_idx == -1:
            continue

        headers = re.split(r'\t|\s{2,}', lines[header_idx].strip())
        headers = [h.strip() for h in headers if h.strip()]
        mag_col = next(i for i, h in enumerate(headers) if 'Mag Field' in h)
        angle_col = next(i for i, h in enumerate(headers) if 'Angle' in h)
        lockin_cols = [i for i, h in enumerate(headers) if 'Lock-in' in h and ('x' in h or 'X' in h)]
        lockin_col = lockin_cols[1] if len(lockin_cols) >= 2 else lockin_cols[0]

        all_data = []
        for line in lines[header_idx+1:]:
            cols = re.split(r'\t|\s{2,}', line.strip())
            cols = [c.strip() for c in cols if c.strip()]
            if len(cols) <= max(mag_col, angle_col, lockin_col): continue
            try:
                mag = float(cols[mag_col])
                angle = float(cols[angle_col])
                lockin = float(cols[lockin_col])
                all_data.append((mag, angle, lockin))
            except: continue
        if not all_data: continue

        # 分组
        mags = sorted([d[0] for d in all_data])
        groups = []
        cur = [mags[0]]
        for m in mags[1:]:
            if abs(m - cur[-1]) <= 2:
                cur.append(m)
            else:
                groups.append(cur)
                cur = [m]
        if cur: groups.append(cur)

        processed = OrderedDict()
        for g in groups:
            avg = round(np.mean(g), 0)
            group_data = [d for d in all_data if d[0] in g]
            last = OrderedDict()
            for m,a,l in group_data:
                last[a] = l
            angle_list = list(last.items())
            if sort_angle: angle_list.sort(key=lambda x: x[0])
            processed[avg] = angle_list

        if not enable_fitting:
            # 仅输出处理后数据表格
            max_rows = max(len(v) for v in processed.values())
            rows = []
            for i in range(max_rows):
                row = []
                for mag in processed.keys():
                    if i < len(processed[mag]):
                        a,l = processed[mag][i]
                        row.extend([mag, a, l])
                    else:
                        row.extend([np.nan, np.nan, np.nan])
                rows.append(row)
            df_out = pd.DataFrame(rows)
            out_buf = io.BytesIO()
            df_out.to_excel(out_buf, index=False)
            out_buf.seek(0)
            all_results.append((filename, out_buf, None, None))
            continue

        # 拟合
        fit_results = []
        initial_guess = [-90, 0, 0, 0, 0, 0, 0]
        for mag, a_list in processed.items():
            xs = np.array([a for a,_ in a_list if not np.isnan(a) and not np.isnan(_)])
            ys = np.array([l*1e6 for a,l in a_list if not np.isnan(a) and not np.isnan(l)])
            if len(xs) < 7:
                fit_results.append({'Mag': mag, 'A': np.nan, 'B': np.nan, 'C': np.nan,
                                     'D': np.nan, 'E': np.nan, 'F': np.nan, 'G': np.nan, 'R²': np.nan})
                continue
            try:
                popt, _ = curve_fit(fit_func, xs, ys, p0=initial_guess, maxfev=10000)
                y_pred = fit_func(xs, *popt)
                r2 = calc_r2(ys, y_pred)
                fit_results.append({'Mag': mag, 'A': popt[0], 'B': popt[1], 'C': popt[2],
                                     'D': popt[3], 'E': popt[4], 'F': popt[5], 'G': popt[6], 'R²': r2})
            except:
                fit_results.append({'Mag': mag, 'A': np.nan, 'B': np.nan, 'C': np.nan,
                                     'D': np.nan, 'E': np.nan, 'F': np.nan, 'G': np.nan, 'R²': np.nan})

        df_fit = pd.DataFrame(fit_results)
        out_buf = io.BytesIO()
        df_fit.to_excel(out_buf, index=False)
        out_buf.seek(0)

        # 合并图
        if generate_plot and len(processed) > 0:
            num_groups = len(processed)
            cols_plot = min(3, num_groups)
            rows_plot = (num_groups + cols_plot - 1) // cols_plot
            fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(cols_plot*8, rows_plot*6))
            if num_groups == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            colors = plt.cm.viridis(np.linspace(0, 1, num_groups))
            for idx, (mag, a_list) in enumerate(processed.items()):
                xs = np.array([a for a,_ in a_list])
                ys = np.array([l*1e6 for _,l in a_list])
                ax = axes[idx]
                ax.scatter(xs, ys, color=colors[idx], s=50)
                fit_row = next((r for r in fit_results if r['Mag'] == mag), None)
                if fit_row and not np.isnan(fit_row['R²']):
                    x_fit = np.linspace(min(xs), max(xs), 200)
                    popt = [fit_row['A'], fit_row['B'], fit_row['C'], fit_row['D'], fit_row['E'], fit_row['F'], fit_row['G']]
                    y_fit = fit_func(x_fit, *popt)
                    ax.plot(x_fit, y_fit, 'r-', lw=2)
                ax.set_title(f'Mag = {mag} Oe')
                ax.grid(True, linestyle='--')
            plt.tight_layout()
            plot_buf = io.BytesIO()
            fig.savefig(plot_buf, format='png', dpi=150)
            plot_buf.seek(0)
            plt.close(fig)
            all_results.append((filename, out_buf, df_fit, plot_buf))
        else:
            all_results.append((filename, out_buf, None, None))

    return all_results

# 6. AHE Y 轴自动偏移校正
def ahe_offset(uploaded_file):
    content = uploaded_file.read().decode('utf-8')
    lines = content.strip().split('\n')
    headers = []
    x_data, y_data = [], []
    first_data = True
    for line in lines:
        parts = line.split()
        if not parts: continue
        if first_data:
            try:
                float(parts[0])
                x_data.append(float(parts[0]))
                if len(parts) >= 2: y_data.append(float(parts[1]))
                first_data = False
            except:
                headers = parts
                first_data = False
                continue
        else:
            try:
                x_data.append(float(parts[0]))
                if len(parts) >= 2: y_data.append(float(parts[1]))
            except: continue
    if not x_data:
        return None
    x = np.array(x_data)
    y = np.array(y_data)
    offset = (np.max(y) + np.min(y)) / 2
    y_shifted = y - offset

    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, 'b-')
    ax1.set_title('Original')
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0); plt.close()

    fig2, ax2 = plt.subplots()
    ax2.plot(x, y_shifted, 'r-')
    ax2.set_title('Shifted')
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0); plt.close()

    data_buf = io.BytesIO()
    header_line = f"# Offset: {offset:.6f}\n"
    if headers:
        header_line += ' '.join(headers) + ' Shifted_Y\n'
    else:
        header_line += 'X Y Shifted_Y\n'
    data_buf.write(header_line.encode())
    for i in range(len(x)):
        data_buf.write(f"{x[i]} {y[i]} {y_shifted[i]}\n".encode())
    data_buf.seek(0)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, 'w') as zf:
        zf.writestr('original.png', buf1.getvalue())
        zf.writestr('shifted.png', buf2.getvalue())
        zf.writestr('shifted_data.txt', data_buf.getvalue())
    zip_buf.seek(0)
    return zip_buf

# ---------------- 脚本注册表（与界面联动） ----------------
SCRIPTS = {
    "📊 谐波数据处理": [
        {
            "name": "谐波斜率曲率分析",
            "func": "harmonic_slope",
            "description": """上传 .txt 文件，提取电流值，进行二次/线性拟合，计算比值并绘图""",
            "upload_multiple": True,
            "file_type": ["txt"],
            "params": [
                {"key": "x_min", "label": "X轴下限", "type": "float", "default": -500.0},
                {"key": "x_max", "label": "X轴上限", "type": "float", "default": 500.0}
            ]
        },
        {
            "name": "谐波提取上升/下降段",
            "func": "extract_segments",
            "description": "上传单个扫描文件，提取上升段和下降段",
            "upload_multiple": False,
            "file_type": ["txt"],
            "params": [
                {"key": "range_min", "label": "范围下限", "type": "float", "default": -0.5},
                {"key": "range_max", "label": "范围上限", "type": "float", "default": 0.5},
                {"key": "tolerance", "label": "容差", "type": "float", "default": 1e-6}
            ]
        },
        {
            "name": "单侧谐波跳点去除",
            "func": "harmonic_remove_spikes",
            "description": "上传谐波测量文件，自动剔除异常跳点",
            "upload_multiple": True,
            "file_type": ["txt"],
            "params": [
                {"key": "z_threshold", "label": "Z-score 阈值", "type": "float", "default": 2.0},
                {"key": "diff_threshold", "label": "第二谐波差分阈值", "type": "float", "default": 2e-7},
                {"key": "use_first", "label": "启用第一谐波辅助 (True/False)", "type": "str", "default": "True"},
                {"key": "first_diff_threshold", "label": "第一谐波差分阈值", "type": "float", "default": 2e-5},
                {"key": "detect_sign", "label": "检测正负跳变 (True/False)", "type": "str", "default": "True"},
                {"key": "sign_threshold", "label": "正负跳变阈值", "type": "float", "default": 1e-8},
                {"key": "suffix", "label": "输出文件后缀", "type": "str", "default": "_cleaned"}
            ]
        }
    ],
    "📈 通用数据处理": [
        {
            "name": "Excel 批量合并（跳过行）",
            "func": "excel_merge",
            "description": "上传 Excel 文件，跳过 N 行后合并第3、4列",
            "upload_multiple": True,
            "file_type": ["xlsx", "xls"],
            "params": [
                {"key": "skip_rows", "label": "跳过行数", "type": "int", "default": 530}
            ]
        }
    ],
    "🧪 PPMS 数据处理": [
        {
            "name": "PPMS 数据拟合（角度扫描）",
            "func": "ppms_fit",
            "description": "上传 PPMS 角度扫描数据，按 Mag 分组进行七参数拟合",
            "upload_multiple": True,
            "file_type": ["txt"],
            "params": [
                {"key": "sort_angle", "label": "按Angle排序 (True/False)", "type": "str", "default": "True"},
                {"key": "enable_fitting", "label": "开启拟合 (True/False)", "type": "str", "default": "True"},
                {"key": "generate_plot", "label": "生成合并图 (True/False)", "type": "str", "default": "True"}
            ]
        }
    ],
    "⚡ AHE 数据处理": [
        {
            "name": "AHE Y 轴自动偏移校正",
            "func": "ahe_offset",
            "description": "上传 AHE 数据，自动归零偏移校正",
            "upload_multiple": False,
            "file_type": ["txt"],
            "params": []
        }
    ]
}

# ---------------- 界面交互逻辑 ----------------
cat = st.selectbox("📌 选择任务类别", list(SCRIPTS.keys()))
scripts = SCRIPTS[cat]
script_names = [s["name"] for s in scripts]
selected_name = st.selectbox("📋 选择具体脚本", script_names)
selected = next(s for s in scripts if s["name"] == selected_name)

with st.expander("📖 脚本说明"):
    st.markdown(selected["description"])

# 文件上传
if selected["upload_multiple"]:
    uploaded_files = st.file_uploader("📂 上传文件", type=selected["file_type"], accept_multiple_files=True)
else:
    uploaded_file = st.file_uploader("📂 上传文件", type=selected["file_type"], accept_multiple_files=False)
    uploaded_files = [uploaded_file] if uploaded_file else []

# 额外参数
param_values = {}
if selected["params"]:
    st.markdown("### ⚙️ 参数设置")
    cols = st.columns(2)
    for i, p in enumerate(selected["params"]):
        with cols[i % 2]:
            if p["type"] == "float":
                param_values[p["key"]] = st.number_input(p["label"], value=p["default"], format="%g")
            elif p["type"] == "int":
                param_values[p["key"]] = st.number_input(p["label"], value=p["default"], step=1)
            else:
                param_values[p["key"]] = st.text_input(p["label"], value=p["default"])

# 运行按钮
if st.button("▶️ 开始处理", use_container_width=True):
    if not uploaded_files:
        st.error("请先上传文件")
    else:
        with st.spinner("正在处理..."):
            func_name = selected["func"]
            if func_name == "harmonic_slope":
                df, img_buf = harmonic_slope_process(uploaded_files, param_values["x_min"], param_values["x_max"])
                if not df.empty:
                    st.success("处理完成")
                    st.dataframe(df)
                    out_excel = io.BytesIO()
                    df.to_excel(out_excel, index=False)
                    out_excel.seek(0)
                    st.download_button("⬇️ 下载 Excel", out_excel, "result.xlsx")
                    st.image(img_buf)
                    img_buf.seek(0)
                    st.download_button("⬇️ 下载图片", img_buf, "plot.png", mime="image/png")
                else:
                    st.warning("未能提取有效数据")
            elif func_name == "extract_segments":
                if uploaded_files:
                    asc, desc = extract_segments(uploaded_files[0], param_values["range_min"], param_values["range_max"], param_values["tolerance"])
                    if asc:
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, 'w') as zf:
                            zf.writestr("ascending.txt", asc)
                            zf.writestr("descending.txt", desc)
                        zip_buf.seek(0)
                        st.success("提取完成")
                        st.download_button("⬇️ 下载结果 (ZIP)", zip_buf, "segments.zip")
                    else:
                        st.error("提取失败，请检查范围设置")
            elif func_name == "harmonic_remove_spikes":
                use_first = param_values["use_first"].lower() == "true"
                detect_sign = param_values["detect_sign"].lower() == "true"
                zip_buf = harmonic_remove_spikes(uploaded_files, param_values["z_threshold"], param_values["diff_threshold"],
                                                use_first, param_values["first_diff_threshold"], detect_sign,
                                                param_values["sign_threshold"], param_values["suffix"])
                if zip_buf:
                    st.success("清洗完成")
                    st.download_button("⬇️ 下载清洗文件 (ZIP)", zip_buf, "cleaned.zip")
                else:
                    st.warning("无文件被处理")
            elif func_name == "excel_merge":
                out_buf = excel_merge(uploaded_files, param_values["skip_rows"])
                if out_buf:
                    st.success("合并完成")
                    st.download_button("⬇️ 下载合并 Excel", out_buf, "merged.xlsx")
                else:
                    st.warning("无法合并，请检查文件")
            elif func_name == "ppms_fit":
                sort_angle = param_values["sort_angle"].lower() == "true"
                enable_fitting = param_values["enable_fitting"].lower() == "true"
                generate_plot = param_values["generate_plot"].lower() == "true"
                results = ppms_fit(uploaded_files, sort_angle, enable_fitting, generate_plot)
                if results:
                    st.success(f"处理了 {len(results)} 个文件")
                    for fname, excel_buf, fit_df, plot_buf in results:
                        st.markdown(f"**{fname}**")
                        st.download_button(f"⬇️ {fname} 拟合参数 Excel", excel_buf, f"{fname}_fit.xlsx")
                        if plot_buf:
                            st.image(plot_buf)
                            plot_buf.seek(0)
                            st.download_button(f"⬇️ {fname} 拟合图", plot_buf, f"{fname}_plot.png", mime="image/png")
                else:
                    st.warning("无有效文件处理")
            elif func_name == "ahe_offset":
                if uploaded_files:
                    zip_buf = ahe_offset(uploaded_files[0])
                    if zip_buf:
                        st.success("偏移校正完成")
                        st.download_button("⬇️ 下载结果 (ZIP)", zip_buf, "ahe_results.zip")
                    else:
                        st.error("处理失败")
            else:
                st.error("未知功能")

# ---------------- 页脚 ----------------
st.markdown("---")
st.markdown("<center style='color: #888;'>磁性测量数据处理平台 v3.0 | 设计人：王赛赛</center>", unsafe_allow_html=True)
