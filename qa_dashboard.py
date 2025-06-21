import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
from prophet import Prophet
import matplotlib.pyplot as plt

st.markdown("<h4 style='text-align: right; color: gray;'>Created by <b>Anurag Sharma</b></h4>", unsafe_allow_html=True)

# Set layout
st.set_page_config(page_title="QA Dashboard", layout="wide")

# Dark/Light mode toggle
# Enforce Dark Theme only
st.markdown("""
    <style>
    body, .stApp, .css-1d391kg { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1c1f26; }
    .stDataFrame, .element-container { background-color: #1c1f26 !important; color: white !important; }
    .sidebar-content, .stRadio, .stMultiSelect, .stSelectbox, .stNumberInput { color: white !important; background-color: #1c1f26 !important; }
    </style>
""", unsafe_allow_html=True)


st.title("📊 QA Error Tracking Dashboard")

# Upload Excel file
uploaded_file = st.file_uploader("Upload your QA Audit File (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = [col.strip() for col in df.columns]
        df['Processed Date'] = pd.to_datetime(df['Processed Date'], errors='coerce')
        df['Month'] = df['Processed Date'].dt.strftime('%B')

        # Ensure Quality Score is numeric
        df['Quality Score'] = pd.to_numeric(df['Quality Score'], errors='coerce')

        # Calculate Passed and Failed safely (>=100 is Passed)
        df['Passed'] = df['Quality Score'].apply(lambda x: 1 if pd.notnull(x) and x >= 100 else 0)
        df['Failed'] = df['Quality Score'].apply(lambda x: 1 if pd.notnull(x) and x < 100 else 0)

        # Sidebar Filters
        st.sidebar.header("📂 Filters")
        qa_names = st.sidebar.multiselect("Select QA (Audit By)", df['Audit By'].unique(), default=df['Audit By'].unique())
        months = st.sidebar.multiselect("Select Month(s)", df['Month'].unique(), default=df['Month'].unique())

        df_filtered = df.copy()
        if qa_names:
            df_filtered = df_filtered[df_filtered['Audit By'].isin(qa_names)]
        if months:
            df_filtered = df_filtered[df_filtered['Month'].isin(months)]


        tab1, tab2, tab3, tab4 = st.tabs(["📌 Summary", "📊 Charts", "🗂 Drilldowns", "📋 Full Data"])

        with tab1:
            st.markdown("## 📌 Dashboard Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Audits", len(df_filtered))
            with col2:
                st.metric("Total Errors", int(df_filtered['Failed'].sum()))
            with col3:
                pass_rate = (df_filtered['Passed'].sum() / len(df_filtered)) * 100 if len(df_filtered) else 0
                st.metric("Pass Rate", f"{pass_rate:.2f}%")
            with col4:
                most_active = df_filtered['Audit By'].value_counts().idxmax()
                st.metric("Most Active QA", most_active)
            with col5:
                top_error_qa = df_filtered[df_filtered['Failed'] == 1]['Audit By'].value_counts().idxmax()
                st.metric("Most Errors QA", top_error_qa)

        with tab2:
            st.markdown("### 🧑‍💼 QA-wise Pass vs Error")
            qa_summary = df_filtered.groupby('Audit By')[['Passed', 'Failed']].sum().reset_index()
            qa_summary_melted = pd.melt(qa_summary, id_vars='Audit By', value_vars=['Passed', 'Failed'], var_name='Status', value_name='Count')
            fig1 = px.bar(qa_summary_melted, x='Audit By', y='Count', color='Status',
                          barmode='stack', title="QA-wise Pass vs Error",
                          color_discrete_map={'Passed': '#28a745', 'Failed': '#dc3545'})
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("### 🧭 Feedback Type Distribution")
            if 'Feedback type' in df_filtered.columns:
                feedback_counts = df_filtered['Feedback type'].value_counts().reset_index()
                feedback_counts.columns = ['Feedback Type', 'Count']
                fig2 = px.pie(feedback_counts, names='Feedback Type', values='Count',
                              title="Feedback Type Distribution", hole=0.4)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 📈 Forecasting Quality Score Trend")
            forecast_df = df_filtered[['Processed Date', 'Quality Score']].dropna()
            forecast_df = forecast_df.rename(columns={'Processed Date': 'ds', 'Quality Score': 'y'})

            if not forecast_df.empty:
                model = Prophet()
                model.fit(forecast_df)
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                fig4 = px.line(forecast, x='ds', y='yhat', title='Forecasted Quality Score (Next 30 Days)')
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Not enough data available for forecasting.")

        with tab3:
            selected_qa = st.selectbox("🔎 Select QA to view their details:", options=qa_summary['Audit By'].unique())
            drilldown_df = df_filtered[df_filtered['Audit By'] == selected_qa]

            st.markdown(f"### 🧾 Detailed Records for: {selected_qa}")
            st.dataframe(drilldown_df.reset_index(drop=True), use_container_width=True, height=300)

            st.markdown("### 🔍 User-wise Errors (Interactive Filter)")
            user_errors = df_filtered[df_filtered['Failed'] == 1]['User Name'].value_counts().reset_index()
            user_errors.columns = ['User Name', 'Error Count']
            fig3 = px.bar(user_errors, x='User Name', y='Error Count', color='Error Count',
                          title="User-wise Error Count", text='Error Count')
            fig3.update_traces(textposition='outside')
            fig3.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig3, use_container_width=True)

            selected_user = st.selectbox("🔍 Filter Detailed Records by User Name:", options=user_errors['User Name'])
            user_detail_df = df_filtered[df_filtered['User Name'] == selected_user]
            st.markdown(f"### 🗂️ Records for User: {selected_user}")
            st.dataframe(user_detail_df.reset_index(drop=True), use_container_width=True)
            
            
            # ✅ Add Quality Score trend for the selected user
            st.markdown(f"### 📈 Quality Score Trend for {selected_user}")
            user_trend_df = user_detail_df[['Processed Date', 'Quality Score', 'Comments']].dropna()

            if not user_trend_df.empty:
                user_trend_df = user_trend_df.sort_values('Processed Date')
                user_line = px.line(
                    user_trend_df,
                    x='Processed Date',
                    y='Quality Score',
                    title=f"{selected_user}'s Quality Score Over Time",
                    hover_data=['Comments'] if 'Comments' in user_trend_df.columns else None
                )
                user_line.update_layout(template="plotly_dark")
                st.plotly_chart(user_line, use_container_width=True)

                st.markdown("### 🧾 Detailed Trend Records")
                st.dataframe(user_trend_df.reset_index(drop=True), use_container_width=True)
            else:
                st.info("No Quality Score records found for selected user.")
                

        with tab4:
            st.markdown("### 📋 QA Audit Data")
            page_size = 20
            total_pages = (len(df_filtered) - 1) // page_size + 1
            current_page = st.number_input("Page", 1, total_pages, 1)
            start_idx = (current_page - 1) * page_size
            end_idx = start_idx + page_size
            st.dataframe(df_filtered.iloc[start_idx:end_idx], use_container_width=True)

            # Download filtered data
            excel_buffer = io.BytesIO()
            df_filtered.to_excel(excel_buffer, index=False)
            st.download_button("📥 Download Filtered Data", data=excel_buffer.getvalue(), file_name="qa_filtered_data.xlsx")

            # Download full sheet with demographics
            export_buffer = io.BytesIO()
            df.to_excel(export_buffer, index=False)
            st.download_button("📤 Download Complete Dataset with Demographics", data=export_buffer.getvalue(), file_name="qa_complete_data.xlsx")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
    
st.markdown("""
    <hr style="border-top: 1px solid #bbb;">
    <div style="text-align: center; color: gray;">
        © 2025 Created by <strong>Anurag Sharma</strong> | All rights reserved.
    </div>
""", unsafe_allow_html=True)
