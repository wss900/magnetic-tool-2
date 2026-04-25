import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import zipfile

st.set_page_config(page_title="AHE Y 轴偏移校正", page_icon="⚡")
st.title("⚡ AHE Y 轴自动偏移校正")

st.markdown("""
**功能**：上传 AHE 测量数据文件（两列：X 坐标与 Y 信号），自动计算偏移量 `(max_y + min_y)/2`，
将所有 Y 值减去偏移量，使信号中心归零。输出原始图、平移后图、处理后的数据文件。
- 输出打包为 ZIP 下载
""")

uploaded_file = st.file_uploader("📂 选择或拖拽一个 .txt 文件", type="txt")

# 输出图像文件名可选
col1, col2 = st.columns(2)
with col1:
    orig_img_name = st.text_input("原始图文件名", value="ahe_original.png")
with col2:
    shift_img_name = st.text_input("平移图文件名", value="ahe_shifted.png")

if st.button("▶️ 开始处理", use_container_width=True):
    if not uploaded_file:
        st.error("请上传一个 .txt 文件")
    else:
        with st.spinner("正在处理..."):
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')
            headers = []
            x_data = []
            y_data = []
            line_idx = 0
            for line in lines:
                line_idx += 1
                parts = line.split()
                if not parts:
                    continue
                if line_idx == 1:
                    # 尝试判断第一行是否为表头（非数字）
                    try:
                        float(parts[0])
                        # 是数字，当作数据
                        x_data.append(float(parts[0]))
                        if len(parts) >= 2:
                            y_data.append(float(parts[1]))
                    except ValueError:
                        # 是表头
                        headers = parts
                        continue
                else:
                    try:
                        x_data.append(float(parts[0]))
                        if len(parts) >= 2:
                            y_data.append(float(parts[1]))
                    except ValueError:
                        continue

            if not x_data:
                st.error("未提取到有效数据")
            else:
                x = np.array(x_data)
                y = np.array(y_data)
                y_min = np.min(y)
                y_max = np.max(y)
                offset = (y_max + y_min) / 2
                shifted_y = y - offset

                # 生成原始图
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                ax1.plot(x, y, 'b-', linewidth=2)
                ax1.set_title('Original AHE Data')
                ax1.set_xlabel(headers[0] if len(headers) > 0 else 'X')
                ax1.set_ylabel(headers[1] if len(headers) > 1 else 'Y')
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format='png', dpi=150)
                buf1.seek(0)
                plt.close(fig1)

                # 生成平移图
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                ax2.plot(x, shifted_y, 'r-', linewidth=2)
                ax2.set_title('Shifted AHE Data')
                ax2.set_xlabel(headers[0] if len(headers) > 0 else 'X')
                ax2.set_ylabel('Shifted Y')
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format='png', dpi=150)
                buf2.seek(0)
                plt.close(fig2)

                # 生成处理后的数据文件
                data_buf = io.BytesIO()
                header_line = f"# Offset: {offset:.6f} (method: (max+min)/2)\n"
                header_line += f"# Min Y: {y_min:.6f}  Max Y: {y_max:.6f}\n"
                if headers:
                    header_line += ' '.join(headers) + ' Shifted_Y\n'
                else:
                    header_line += 'X Y Shifted_Y\n'
                data_buf.write(header_line.encode('utf-8'))
                for i in range(len(x)):
                    data_buf.write(f"{x[i]} {y[i]} {shifted_y[i]}\n".encode('utf-8'))
                data_buf.seek(0)

                # 打包 ZIP
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr(orig_img_name, buf1.getvalue())
                    zf.writestr(shift_img_name, buf2.getvalue())
                    zf.writestr("shifted_" + uploaded_file.name, data_buf.getvalue())
                zip_buf.seek(0)

                st.success(f"处理完成！偏移量 = {offset:.6f}")
                st.download_button(
                    label="⬇️ 下载处理结果 (ZIP)",
                    data=zip_buf,
                    file_name="ahe_results.zip",
                    mime="application/zip"
                )
