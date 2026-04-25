import streamlit as st
import pandas as pd
import numpy as np
import io
import os

st.set_page_config(page_title="Excel 批量合并", page_icon="📊")
st.title("📊 Excel 批量合并（跳过行）")

st.markdown("""
**功能**：上传多个 Excel 文件，提取第3列和第4列数据，跳过前 N 行后合并为一个 Excel 表格。
- 每个文件的两列数据以文件名作为列名前缀
- 输出为一个合并后的 Excel 文件，可直接下载
""")

uploaded_files = st.file_uploader("📂 选择或拖拽 Excel 文件", type=["xlsx", "xls"], accept_multiple_files=True)

skip_rows = st.number_input("跳过前 N 行", min_value=0, value=530, step=10)

if st.button("▶️ 开始合并", use_container_width=True):
    if not uploaded_files:
        st.error("请至少上传一个 Excel 文件")
    else:
        data_list = []
        file_names = []
        with st.spinner("正在处理..."):
            for uploaded in uploaded_files:
                try:
                    df = pd.read_excel(uploaded)
                except Exception as e:
                    st.warning(f"读取文件 {uploaded.name} 失败: {e}")
                    continue

                if df.empty or df.shape[1] < 4:
                    st.warning(f"文件 {uploaded.name} 列数不足4列，跳过")
                    continue

                x_col = df.columns[2]
                y_col = df.columns[3]

                x = pd.to_numeric(df[x_col], errors='coerce')
                y = pd.to_numeric(df[y_col], errors='coerce')

                base_name = os.path.splitext(uploaded.name)[0]
                temp = pd.DataFrame({
                    f"{base_name}_X": np.abs(x),
                    f"{base_name}_Y": np.abs(y)
                }).dropna()

                if len(temp) > skip_rows:
                    temp = temp.iloc[skip_rows:].reset_index(drop=True)
                    data_list.append(temp)
                    file_names.append(base_name)
                else:
                    st.warning(f"文件 {uploaded.name} 数据行数不足（跳过 {skip_rows} 行后无数据）")

        if data_list:
            all_data = pd.concat(data_list, axis=1)
            # 输出到 BytesIO
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                all_data.to_excel(writer, index=False)
            output.seek(0)

            st.success(f"合并完成！共处理 {len(file_names)} 个文件")
            st.download_button(
                label="⬇️ 下载合并后的 Excel",
                data=output,
                file_name="merged_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("没有文件被成功处理")
