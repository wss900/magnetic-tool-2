import streamlit as st
import io
import zipfile

st.set_page_config(page_title="谐波提取上升/下降段", page_icon="⬆️⬇️")
st.title("⬆️⬇️ 谐波提取上升/下降段")

st.markdown("""
**功能**：上传一个完整的磁场扫描数据文件（.txt），自动提取上升段（最小值→最大值）和下降段（最大值→最小值），
分别保存为两个文件并打包下载。
**输入**：单列或多列数据，以第一列为判断标准。
**参数**：上升/下降的始末范围与浮点比较容差。
""")

uploaded_file = st.file_uploader("📂 选择或拖拽一个 .txt 文件", type="txt")

col1, col2, col3 = st.columns(3)
with col1:
    range_min = st.number_input("范围下限", value=-0.5, format="%.3f")
with col2:
    range_max = st.number_input("范围上限", value=0.5, format="%.3f")
with col3:
    tolerance = st.number_input("浮点容差", value=1e-6, format="%.1e", step=1e-7)

if st.button("▶️ 开始提取", use_container_width=True):
    if not uploaded_file:
        st.error("请上传一个 .txt 文件")
    else:
        with st.spinner("正在提取..."):
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')

            # 提取上升段
            asc_lines = []
            found_min = False
            found_max = False
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                try:
                    val = float(parts[0])
                except ValueError:
                    continue
                if not found_min:
                    if abs(val - range_min) < tolerance:
                        found_min = True
                        asc_lines.append(line)
                        continue
                else:
                    asc_lines.append(line)
                    if abs(val - range_max) < tolerance:
                        found_max = True
                        break

            if not found_min or not found_max:
                st.error("未找到完整的上升段，请检查范围设置或文件格式")
            else:
                # 提取下降段（从上升段结束之后开始）
                try:
                    after_idx = lines.index(asc_lines[-1]) + 1
                except ValueError:
                    after_idx = len(lines)

                desc_lines = []
                found_max2 = False
                found_min2 = False
                for i in range(after_idx, len(lines)):
                    line = lines[i]
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        val = float(parts[0])
                    except ValueError:
                        continue
                    if not found_max2:
                        if abs(val - range_max) < tolerance:
                            found_max2 = True
                            desc_lines.append(line)
                            continue
                    else:
                        desc_lines.append(line)
                        if abs(val - range_min) < tolerance:
                            found_min2 = True
                            break

                if not found_max2 or not found_min2:
                    st.error("未找到完整的下降段，请检查数据")
                else:
                    # 生成两个 BytesIO
                    asc_buf = io.BytesIO('\n'.join(asc_lines).encode('utf-8'))
                    desc_buf = io.BytesIO('\n'.join(desc_lines).encode('utf-8'))
                    asc_buf.seek(0)
                    desc_buf.seek(0)

                    # 打包 ZIP
                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr('ascending.txt', asc_buf.getvalue())
                        zf.writestr('descending.txt', desc_buf.getvalue())
                    zip_buf.seek(0)

                    st.success("提取完成！上升段 %d 行，下降段 %d 行" % (len(asc_lines), len(desc_lines)))
                    st.download_button(
                        label="⬇️ 下载上升段+下降段 (ZIP)",
                        data=zip_buf,
                        file_name="segments.zip",
                        mime="application/zip"
                    )
