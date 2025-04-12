import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Customer Segmentation", layout="centered")
df_total = pd.read_csv("dataframe_total.csv")
product_df = pd.read_csv("df_product_sales.csv")
category_df = pd.read_csv("df_category_sales.csv")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Menu",
    ["Home", "Data Insight", "Segmentation"]
)

# Trang Home
if menu == "Home":
    st.title("Customer Segmentation Project")
    st.write("Welcome to the Customer Segmentation Dashboard!")
    st.image("Fruit.jpg")
    st.header("Business Understanding:")
    st.write("Store X mainly sells essential products to customers such as vegetables, fruits, meat, fish, eggs, milk, soft drinks, etc. The store's customers are retail customers.")
    st.write("The Customer Segmentation project uses the RFM (Recency, Frequency, Monetary) analysis method to divide customers into groups to serve different marketing strategies.")

    st.write("  Recency: Number of days since the last purchase")
    st.write("  Frequency: Number of purchases")
    st.write("  Monetary: Total order value")
    st.write("Combining the RFM method and the KMeans algorithm, the customer segmentation system helps increase efficiency in personalizing customer care and retention, while helping businesses increase revenue.")


# Trang Data Insight
elif menu == "Data Insight":
    st.title("Data Insight")
    
    st.subheader("Xem trước dữ liệu RFM")
    
    num_rows = st.number_input(
        "Input number of row", 
        min_value=2, 
        max_value=100, 
        value=5,  # Giá trị mặc định
        step=1
    )
    
    st.dataframe(df_total.head(num_rows))
    st.write("Data recored from 1/1/2024 to 30/12/2015")
    st.subheader("""There are total:
                 
        ->11 Product Categories
                 
        ->167 Products
                 
        ->3898 Customers
                 
        ->77380 Products Sold""")
    st.subheader("""With:

        ->Whole Milk is the best selling product
                 
        ->Fresh Food is the best selling category
                 
        ->Total Income: 332159.63$
                 
        ->May is the month with most sales in 2014
                 
        ->August is the month with most sales in 2015 """)
    st.subheader("📦 Top Seller Products")

    num_products = st.number_input(
        "Choose number of products to see", 
        min_value=2, 
        max_value=len(product_df), 
        value=5
    )

    top_products = product_df.sort_values(by="counts", ascending=False).head(num_products)

    fig_product = px.bar(top_products, 
                        x="productName", 
                        y="counts",
                        color="counts",
                        title=f"Top {num_products} best seller products",
                        labels={"product_name": "Sản phẩm", "counts": "Số lượng bán"})

    st.plotly_chart(fig_product, use_container_width=True)

    st.subheader("🛍️ Top Seller Category")

    num_categories = st.number_input(
        "Choose number of catehory to see", 
        min_value=2, 
        max_value=len(category_df), 
        value=5
    )

    top_categories = category_df.sort_values(by="count_cat", ascending=False).head(num_categories)

    fig_category = px.bar(top_categories, 
                        x="Category", 
                        y="count_cat",
                        color="count_cat",
                        title=f"Top {num_categories} danh mục sản phẩm bán chạy nhất",
                        labels={"Category": "Danh mục", "count_cat": "Số lượng bán"})

    st.plotly_chart(fig_category, use_container_width=True)

    st.write("Kết quả phân cụm: ")
    st.image("output.png")
    st.image("newplot.png")

# Trang Segmentation
elif menu == "Segmentation":
    st.title("🔍 Customer Segmentation")

    # Tải mô hình và scaler
    @st.cache_resource
    def load_model_scaler():
        model = pickle.load(open("kmeans_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return model, scaler

    model, scaler = load_model_scaler()

    # Tải dữ liệu RFM
    rfm_df = pd.read_csv("rfm_output.csv")

    # Lựa chọn kiểu nhập liệu
    input_mode = st.radio("Chọn phương thức dự đoán:", ["🔢 Nhập mã khách hàng", "✍️ Nhập thủ công RFM"])

    if input_mode == "🔢 Nhập mã khách hàng":
        member_id = st.number_input("Nhập mã khách hàng (4 chữ số):", min_value=1000, max_value=9999, step=1)
        if st.button("🔍 Dự đoán"):
            if member_id in rfm_df["Member_number"].values:
                row = rfm_df[rfm_df["Member_number"] == member_id]
                rfm_values = row[["Frequency", "Recency", "Monetary"]].values
                rfm_scaled = scaler.transform(rfm_values)
                cluster = model.predict(rfm_scaled)[0]

                st.success(f"📊 Khách hàng **{member_id}** thuộc **Cluster {cluster}**")

                if cluster == 0:
                    st.info("🟡 Nhóm khách hàng trung thành hoặc chi tiêu nhiều.")
                elif cluster == 1:
                    st.info("🔵 Nhóm khách hàng bình thường.")
                elif cluster == 2:
                    st.info("🔴 Nhóm khách hàng ít hoạt động hoặc mới.")
            else:
                st.warning("❗ Mã khách hàng không tồn tại trong dữ liệu.")

    else:
        st.subheader("✍️ Nhập thông tin RFM của khách hàng:")
        recency = st.number_input("Recency (Số ngày gần nhất mua hàng)", min_value=0)
        frequency = st.number_input("Frequency (Số lần mua hàng)", min_value=0)
        monetary = st.number_input("Monetary (Tổng chi tiêu)", min_value=0.0)

        if st.button("🔎 Dự đoán nhóm khách hàng"):
            rfm_input = np.array([[frequency, recency, monetary]])
            rfm_scaled = scaler.transform(rfm_input)
            cluster = model.predict(rfm_scaled)[0]

            st.success(f"📊 Khách hàng thuộc **Cluster {cluster}**")

            if cluster == 0:
                st.info("🟡 Nhóm khách hàng trung thành hoặc chi tiêu nhiều.")
            elif cluster == 1:
                st.info("🔵 Nhóm khách hàng bình thường.")
            elif cluster == 2:
                st.info("🔴 Nhóm khách hàng ít hoạt động hoặc mới.")

