import streamlit as st
from googleapiclient.discovery import build
from google.cloud import bigquery
import pandas as pd
import os

# ================= CONFIG =================
API_KEY = "AIzaSyC0gDJ5ipTodDrMGHF2-Zg0qMftp_2UY6E"
youtube = build("youtube", "v3", developerKey=API_KEY)

st.set_page_config(page_title="Add Channel", layout="wide")

st.title("➕ Thêm kênh mới")
st.caption("Thêm kênh mới vào hệ thống để phân tích")

# ================= BIGQUERY CLIENT =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
credentials_path = os.path.join(
    BASE_DIR,
    "credentials",
    "project-8fd99edc-9e20-4b82-b43-41fc5f2ccbcd.json"
)
client = bigquery.Client.from_service_account_json(
    credentials_path,
    project="project-8fd99edc-9e20-4b82-b43"
)

# ================= FUNCTIONS =================

def get_channel_by_channel_id(channel_id):
    """Lấy thông tin kênh bằng Channel ID"""
    try:
        request = youtube.channels().list(
            part="statistics,snippet",
            id=channel_id
        )
        response = request.execute()
        
        if response["items"]:
            item = response["items"][0]
            stats = item["statistics"]
            snippet = item["snippet"]
            
            return {
                "channel_id": channel_id,
                "channel_name": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "subscriber_count": int(stats.get("subscriberCount", 0)),
                "total_views": int(stats.get("viewCount", 0)),
                "total_videos": int(stats.get("videoCount", 0))
            }
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")
        return None

def search_channels_by_name(keyword):
    """Tìm kiếm kênh bằng tên"""
    try:
        request = youtube.search().list(
            part="snippet",
            q=keyword,
            type="channel",
            maxResults=10
        )
        response = request.execute()
        channels = []
        
        for item in response.get("items", []):
            channel_id = item["snippet"]["channelId"]
            channel_info = get_channel_by_channel_id(channel_id)
            if channel_info:
                channels.append(channel_info)
        
        return channels
    except Exception as e:
        st.error(f"❌ Lỗi tìm kiếm: {str(e)}")
        return []

def save_channel_to_bigquery(channel_info):
    """Lưu thông tin kênh vào BigQuery"""
    try:
        # Check if table exists
        query = f"""
        SELECT COUNT(*) as count FROM `project-8fd99edc-9e20-4b82-b43.raw.raw_channels`
        WHERE channel_id = '{channel_info["channel_id"]}'
        """
        result = client.query(query).to_dataframe()
        
        if result['count'].values[0] > 0:
            st.warning(f"⚠️ Kênh '{channel_info['channel_name']}' đã tồn tại trong hệ thống!")
            return False
        
        # Insert new channel
        from datetime import datetime
        insert_query = f"""
        INSERT INTO `project-8fd99edc-9e20-4b82-b43.raw.raw_channels`
        (channel_id, channel_name, description, subscriber_count, total_views, total_videos, thumbnail_url, extracted_at)
        VALUES (
            '{channel_info["channel_id"]}',
            '{channel_info["channel_name"].replace("'", "")}',
            '{channel_info["description"].replace("'", "")}',
            {channel_info["subscriber_count"]},
            {channel_info["total_views"]},
            {channel_info["total_videos"]},
            '{channel_info["thumbnail"]}',
            CURRENT_TIMESTAMP()
        )
        """
        
        client.query(insert_query).result()
        return True
    except Exception as e:
        st.error(f"❌ Lỗi lưu dữ liệu: {str(e)}")
        return False

# ================= UI SECTION 1: SEARCH BY NAME =================
st.subheader("🔍 Tìm kiếm theo tên kênh")
col_search = st.columns(1)[0]
search_keyword = col_search.text_input("Nhập tên kênh hoặc từ khóa tìm kiếm")

if st.button("🔎 Tìm kiếm"):
    if search_keyword.strip():
        with st.spinner("Đang tìm kiếm..."):
            channels = search_channels_by_name(search_keyword)
            
            if channels:
                st.success(f"✅ Tìm thấy {len(channels)} kênh!")
                
                for i, channel in enumerate(channels, 1):
                    with st.expander(f"{i}. {channel['channel_name']}"):
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            if channel['thumbnail']:
                                st.image(channel['thumbnail'], width=150)
                        
                        with col2:
                            st.write(f"**Channel ID:** {channel['channel_id']}")
                            st.write(f"**Subscribers:** {channel['subscriber_count']:,}")
                            st.write(f"**Total Views:** {channel['total_views']:,}")
                            st.write(f"**Total Videos:** {channel['total_videos']:,}")
                            st.write(f"**Description:** {channel['description'][:200]}...")
                            
                            if st.button(f"✅ Thêm kênh này", key=f"add_{i}"):
                                if save_channel_to_bigquery(channel):
                                    st.success(f"✅ Đã thêm kênh '{channel['channel_name']}' vào hệ thống!")
                                    st.session_state.selected_channel_id = channel['channel_id']
                                    st.session_state.selected_channel_name = channel['channel_name']
                                    st.rerun()
            else:
                st.warning("❌ Không tìm thấy kênh nào!")
    else:
        st.warning("Hãy nhập từ khóa tìm kiếm!")

# ================= UI SECTION 2: ADD BY CHANNEL ID =================
st.markdown("---")
st.subheader("🔗 Thêm bằng Channel ID")

channel_id_input = st.text_input("Nhập Channel ID (bắt đầu bằng UC...)")

if st.button("➕ Thêm kênh"):
    if channel_id_input.strip():
        with st.spinner("Đang tải thông tin kênh..."):
            channel_info = get_channel_by_channel_id(channel_id_input)
            
            if channel_info:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.image(channel_info['thumbnail'], width=150)
                
                with col2:
                    st.write(f"**Tên kênh:** {channel_info['channel_name']}")
                    st.write(f"**Subscribers:** {channel_info['subscriber_count']:,}")
                    st.write(f"**Total Views:** {channel_info['total_views']:,}")
                    st.write(f"**Total Videos:** {channel_info['total_videos']:,}")
                
                if st.button("✅ Xác nhận thêm kênh"):
                    if save_channel_to_bigquery(channel_info):
                        st.success(f"✅ Đã thêm kênh '{channel_info['channel_name']}' vào hệ thống!")
                        st.session_state.selected_channel_id = channel_info['channel_id']
                        st.session_state.selected_channel_name = channel_info['channel_name']
                        st.rerun()
            else:
                st.error("❌ Không tìm thấy kênh với ID này!")
    else:
        st.warning("Hãy nhập Channel ID!")

# ================= UI SECTION 3: MANAGE CHANNELS =================
st.markdown("---")
st.subheader("📋 Danh sách kênh đã thêm")

try:
    query = """
    SELECT channel_id, channel_name, subscriber_count, total_views, total_videos, extracted_at
    FROM `project-8fd99edc-9e20-4b82-b43.raw.raw_channels`
    ORDER BY extracted_at DESC
    LIMIT 50
    """
    channels_df = client.query(query).to_dataframe()
    
    if not channels_df.empty:
        display_df = channels_df.copy()
        display_df['subscriber_count'] = display_df['subscriber_count'].apply(lambda x: f"{x:,}")
        display_df['total_views'] = display_df['total_views'].apply(lambda x: f"{x:,}")
        display_df['extracted_at'] = pd.to_datetime(display_df['extracted_at']).dt.strftime('%d/%m/%Y %H:%M')
        
        st.dataframe(display_df, width='stretch')
        st.caption(f"📊 Tổng cộng: {len(channels_df)} kênh")
    else:
        st.info("📭 Chưa có kênh nào được thêm vào hệ thống.")
except Exception as e:
    st.warning(f"⚠️ Không thể tải danh sách kênh: {str(e)}")

# ================= FOOTER =================
st.markdown("---")
st.caption("YouTube Analytics Pipeline • Channel Management")
