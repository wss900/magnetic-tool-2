import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import os

st.set_page_config(page_title="单侧谐波跳点去除", page_icon="🧹")
st.title("🧹 单侧谐波跳点去除")

st.markdown("""
**功能**：上传谐波测量文件（三列数据），自动检测并剔除异常跳点，下载清洗后的文件。
检测方法：
- 第二谐波 Z-score 阈值
- 第二谐波差分阈值
- 正负值跳变检测（可选）
- 第一谐波辅助差分检测（可选）
输出：清洗后的文件（打包为 ZIP 下载）
""")

uploaded_files = st.file_uploader("📂 选择或拖拽 .txt 文件", type="txt", accept_multiple_files=True)

# 参数设置
st.markdown("### ⚙️ 检测参数")
col1, col2 = st.columns(2)
with col1:
    z_threshold = st.number_input("Z-score 阈值", value=2.0, step=0.1)
    diff_threshold = st.number_input("第二谐波差分阈值", value=0.0000002, format="%.1e")
    first_diff_threshold = st.number_input("第一谐波差分阈值", value=0.00002, format="%.1e")
with col2:
    use_first = st.selectbox("启用第一谐波辅助", ["True", "False"], index=0)
    detect_sign = st.selectbox("检测正负跳变", ["True", "False"], index=0)
    sign_threshold = st.number_input("正负跳变阈值", value=0.00000001, format="%.1e")

file_suffix = st.text_input("输出文件后缀", value="_cleaned")

if st.button("▶️ 开始清洗", use_container_width=True):
    if not uploaded_files:
        st.error("请至少上传一个 .txt 文件")
    else:
        cleaned_data = {}
        with st.spinner("正在处理..."):
            for uploaded in uploaded_files:
                filename = uploaded.name
                try:
                    content = uploaded.read().decode('utf-8')
                    df = pd.read_csv(io.StringIO(content), sep='\t', header=None)
                except Exception as e:
                    st.warning(f"读取文件 {filename} 失败: {e}")
                    continue

                if df.shape[1] < 3:
                    st.warning(f"文件 {filename} 列数不足3列，跳过")
                    continue

                try:
                    v1 = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
                    v2 = pd.to_numeric(df.iloc[:, 2], errors='coerce').values
                except:
                    st.warning(f"文件 {filename} 数据转换失败")
                    continue

                n = len(v2)
                mask = np.ones(n, dtype=bool)

                # Z-score
                if z_threshold is not None:
                    mean_v2 = np.mean(v2)
                    std_v2 = np.std(v2)
                    if std_v2 > 0:
                        z = np.abs((v2 - mean_v2) / std_v2)
                        mask &= (z < z_threshold)

                # 差分
                if diff_threshold is not None:
                    diff_fwd = np.abs(np.diff(v2, prepend=v2[0]))
                    diff_bwd = np.abs(np.diff(v2, append=v2[-1]))
                    diff_max = np.maximum(diff_fwd, diff_bwd)
                    mask &= (diff_max < diff_threshold)

                # 正负跳变
                if detect_sign == "True":
                    sign_changes = np.zeros(n, dtype=bool)
                    for i in range(1, n):
                        if v2[i] * v2[i-1] < 0 and abs(v2[i] - v2[i-1]) > sign_threshold:
                            sign_changes[i] = True
                            sign_changes[i-1] = True
                    mask &= ~sign_changes

                # 第一谐波辅助
                if use_first == "True":
                    diff_v1 = np.abs(np.diff(v1, prepend=v1[0]))
                    mask &= (diff_v1 < first_diff_threshold)

                df_clean = df[mask].reset_index(drop=True)

                # 保存清洗后文件
                out_buf = io.BytesIO()
                df_clean.to_csv(out_buf, sep='\t', index=False, header=False)
                out_buf.seek(0)

                base, ext = os.path.splitext(filename)
                cleaned_name = f"{base}{file_suffix}{ext}"
                cleaned_data[cleaned_name] = out_buf

        if cleaned_data:
            # 打包成 ZIP 下载
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for name, buf in cleaned_data.items():
                    zf.writestr(name, buf.getvalue())
            zip_buf.seek(0)

            st.success(f"处理完成！共清洗 {len(cleaned_data)} 个文件。")
            st.download_button(
                label="⬇️ 下载清洗后文件 (ZIP)",
                data=zip_buf,
                file_name="cleaned_data.zip",
                mime="application/zip"
            )
        else:
            st.warning("没有文件被成功处理")
