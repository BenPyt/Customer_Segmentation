import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

# Cấu hình trang
st.set_page_config(page_title="Phân Khúc Khách Hàng", layout="centered")
df_total = pd.read_csv("dataframe_total.csv")
product_df = pd.read_csv("df_product_sales.csv")
category_df = pd.read_csv("df_category_sales.csv")


# Menu bên trái
st.sidebar.title("🎓 Thông tin nhóm")

st.sidebar.markdown("""
**👩‍🏫 Giáo viên hướng dẫn:**  
*Khuat Thuy Phuong*

**👨‍💻 Thành viên nhóm:**  
- 🧠 Huỳnh Thiện Tấn: Làm GUI, huấn luyện model  
- 📊 Trần Anh Tuấn: EDA dữ liệu, powerpoint
""")


# Menu bên trái
menu = st.sidebar.selectbox(
    "Menu",
    ["Trang Chủ", "Khám Phá Dữ Liệu", "Phân Khúc"]
)

# Trang Chủ
if menu == "Trang Chủ":
    st.title("Dự Án Phân Khúc Khách Hàng")
    st.write("Chào mừng đến với Bảng Điều Khiển Phân Khúc Khách Hàng!")
    st.image("Fruit.jpg")
    st.header("Hiểu Biết Về Bài Toán Kinh Doanh:")
    st.write("Cửa hàng X chủ yếu bán các mặt hàng thiết yếu như rau củ, trái cây, thịt, cá, trứng, sữa, nước giải khát,... cho khách hàng bán lẻ.")
    st.write("Dự án Phân Khúc Khách Hàng sử dụng phương pháp phân tích RFM (Recency, Frequency, Monetary) để chia khách hàng thành các nhóm khác nhau phục vụ cho các chiến lược marketing.")
    st.write("  - Recency: Số ngày kể từ lần mua gần nhất")
    st.write("  - Frequency: Số lần mua hàng")
    st.write("  - Monetary: Tổng giá trị mua hàng")
    st.write("Việc kết hợp phương pháp RFM với thuật toán KMeans giúp hệ thống phân nhóm khách hàng hoạt động hiệu quả hơn trong việc cá nhân hóa chăm sóc và giữ chân khách hàng, đồng thời giúp doanh nghiệp gia tăng doanh thu.")

# Trang Khám Phá Dữ Liệu
elif menu == "Khám Phá Dữ Liệu":
    st.title("Khám Phá Dữ Liệu")
    
    st.subheader("Tìm Hiểu Dữ Liệu")
    
    num_rows = st.number_input(
        "Nhập số dòng cần xem", 
        min_value=2, 
        max_value=100, 
        value=5,  # Giá trị mặc định
        step=1
    )
    
    st.dataframe(df_total.head(num_rows))
    st.write("Dữ liệu được ghi nhận từ ngày 1/1/2024 đến 30/12/2015")
    st.subheader("""Tổng cộng có:
                 
        ->11 danh mục sản phẩm
                 
        ->167 sản phẩm
                 
        ->3898 khách hàng
                 
        ->77380 sản phẩm đã bán""")
    st.subheader("""Trong đó:

        ->Sản phẩm bán chạy nhất là Whole Milk
                 
        ->Danh mục bán chạy nhất là Fresh Food
                 
        ->Tổng doanh thu: 332159.63$
                 
        ->Tháng có doanh số cao nhất năm 2014 là tháng 5
                 
        ->Tháng có doanh số cao nhất năm 2015 là tháng 8""")
    st.subheader("📦 Top sản phẩm bán chạy nhất")

    num_products = st.number_input(
        "Chọn số lượng sản phẩm muốn xem", 
        min_value=2, 
        max_value=len(product_df), 
        value=5
    )

    top_products = product_df.sort_values(by="counts", ascending=False).head(num_products)

    fig_product = px.bar(top_products, 
                        x="productName", 
                        y="counts",
                        color="counts",
                        title=f"Top {num_products} sản phẩm bán chạy nhất",
                        labels={"product_name": "Sản phẩm", "counts": "Số lượng bán"})

    st.plotly_chart(fig_product, use_container_width=True)

    st.subheader("🛍️ Top danh mục bán chạy")

    num_categories = st.number_input(
        "Chọn số lượng danh mục muốn xem", 
        min_value=2, 
        max_value=len(category_df), 
        value=5
    )

    top_categories = category_df.sort_values(by="count_cat", ascending=False).head(num_categories)

    fig_category = px.bar(top_categories, 
                        x="Category", 
                        y="count_cat",
                        color="count_cat",
                        title=f"Top {num_categories} danh mục bán chạy nhất",
                        labels={"Category": "Danh mục", "count_cat": "Số lượng bán"})

    st.plotly_chart(fig_category, use_container_width=True)

    st.write("Kết quả phân cụm: ")
    st.image("output.png")
    st.image("newplot.png")

# Trang Phân Khúc
elif menu == "Phân Khúc":
    st.title("🔍 Phân Khúc Khách Hàng")

    # Tải model và scaler
    @st.cache_resource
    def load_model_scaler():
        model = pickle.load(open("kmeans_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler

    model, scaler = load_model_scaler()

    # Tải dữ liệu RFM
    rfm_df = pd.read_csv("rfm_output.csv")

    # Chọn phương thức nhập liệu
    input_mode = st.radio("Chọn phương thức dự đoán:", ["🔢 Nhập mã khách hàng", "✍️ Nhập thủ công RFM", "📂 Tải lên file dữ liệu"])


    if input_mode == "🔢 Nhập mã khách hàng":
        member_id = st.number_input("Nhập mã khách hàng (4 chữ số):", min_value=1000, max_value=9999, step=1)
        if st.button("🔍 Dự đoán"):
            if member_id in rfm_df["Member_number"].values:
                row = rfm_df[rfm_df["Member_number"] == member_id]
                rfm_values = row[["Frequency", "Recency", "Monetary"]].values
                rfm_scaled = scaler.transform(rfm_values)
                cluster = model.predict(rfm_scaled)[0]

                st.success(f"📊 Khách hàng **{member_id}** thuộc **Cụm {cluster}**")

                if cluster == 0:
                    st.info("🟡 Khách hàng trung thành, có giá trị cao: Mua hàng khá thường xuyên, chi tiêu cao và mua gần đây. Đây là nhóm nên ưu tiên giữ chân và chăm sóc kỹ.")
                elif cluster == 1:
                    st.info("🔵 Khách hàng thưa thớt, ít giá trị: Mua hàng không thường xuyên, chi tiêu thấp, và lần gần nhất mua cách đây khá lâu (Recency: 309.23). Đây là nhóm cần chiến dịch kích thích quay lại.")
                elif cluster == 2:
                    st.info("🟢 Khách hàng VIP: Rất thường xuyên mua hàng, chi tiêu cao nhất và mua gần đây. Đây là nhóm giữ chân mạnh – cá nhân hóa cao.")
                elif cluster == 3:
                    st.info("🔴 Khách hàng mới hoặc ngủ quên: Mua ít, chi tiêu thấp, nhưng vẫn còn khá gần đây. Đây là nhóm tiềm năng để thúc đẩy mua lại.")
                elif cluster == 4:
                    st.info("⚫ Khách hàng không còn tương tác: Rất lâu không mua, mua ít, chi tiêu rất thấp. Đây là nhóm gần như đã rời bỏ, cân nhắc loại bỏ hoặc remarketing nhẹ.")
            else:
                st.warning("❗ Mã khách hàng không tồn tại trong dữ liệu.")

    elif input_mode == "📂 Tải lên file dữ liệu":
        st.subheader("📥 Tải lên file CSV chứa thông tin RFM")
        uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)

                required_cols = ['Frequency', 'Recency', 'Monetary']
                if all(col in df_upload.columns for col in required_cols):
                    # Scale dữ liệu và dự đoán
                    scaled_input = scaler.transform(df_upload[required_cols])
                    clusters = model.predict(scaled_input)
                    df_upload['Cluster'] = clusters

                    st.success("✅ Dự đoán thành công!")
                    st.dataframe(df_upload)

                    # Tùy chọn tải xuống kết quả
                    csv = df_upload.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Tải kết quả về",
                        data=csv,
                        file_name="du_doan_phan_khuc.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("❗ File phải chứa đầy đủ 3 cột: Frequency, Recency, Monetary.")
            except Exception as e:
                st.error(f"⚠️ Lỗi xử lý file: {e}")


    else:
        st.subheader("✍️ Nhập thông tin RFM của khách hàng:")
        recency = st.number_input("Recency (Số ngày gần nhất mua hàng)", min_value=0)
        frequency = st.number_input("Frequency (Số lần mua hàng)", min_value=0)
        monetary = st.number_input("Monetary (Tổng chi tiêu)", min_value=0.0)

        if st.button("🔎 Dự đoán nhóm khách hàng"):
            rfm_input = np.array([[frequency, recency, monetary]])
            rfm_scaled = scaler.transform(rfm_input)
            cluster = model.predict(rfm_scaled)[0]

            st.success(f"📊 Khách hàng thuộc **Cụm {cluster}**")

            if cluster == 0:
                st.info("🟡 Nhóm khách hàng trung thành hoặc chi tiêu nhiều.")
            elif cluster == 1:
                st.info("🔵 Nhóm khách hàng bình thường.")
            elif cluster == 2:
                st.info("🔴 Nhóm khách hàng ít hoạt động hoặc mới.")
