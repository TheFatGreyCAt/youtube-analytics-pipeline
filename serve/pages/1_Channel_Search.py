import streamlit as st
from googleapiclient.discovery import build

# ================= CONFIG =================
API_KEY = "AIzaSyC0gDJ5ipTodDrMGHF2-Zg0qMftp_2UY6E"

youtube = build("youtube", "v3", developerKey=API_KEY)

st.set_page_config(page_title="YouTube Search", layout="wide")

st.title("🔎 Tìm kiếm YouTube")

query = st.text_input("Nhập từ khóa tìm kiếm")

# ================= SEARCH FUNCTION =================
def search_youtube(keyword):

    request = youtube.search().list(
        q=keyword,
        part="snippet",
        type="channel",
        maxResults=5
    )
    response = request.execute()

    return response["items"]

# ================= SEARCH BUTTON =================
if st.button("Tìm kiếm"):
    if query.strip() == "":
        st.warning("Vui lòng nhập từ khóa.")
    else:
        st.session_state.search_results = search_youtube(query)

# ================= DISPLAY RESULTS =================
if "search_results" in st.session_state:

    for item in st.session_state.search_results:

        channel_id = item["snippet"]["channelId"]
        channel_title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        thumbnail = item["snippet"]["thumbnails"]["high"]["url"]

        # Lấy thống kê kênh
        channel_request = youtube.channels().list(
            part="statistics",
            id=channel_id
        )
        channel_response = channel_request.execute()
        stats = channel_response["items"][0]["statistics"]

        subscribers = stats.get("subscriberCount", "N/A")
        total_views = stats.get("viewCount", "N/A")
        video_count = stats.get("videoCount", "N/A")

        st.markdown("---")

        col1, col2 = st.columns([1, 4])

        with col1:
            st.image(thumbnail, width=120)

        with col2:
            st.subheader(channel_title)
            st.write(f"👥 Subscribers: {subscribers}")
            st.write(f"🎬 Videos: {video_count}")
            st.write(f"👁 Views: {total_views}")
            st.write(description[:200] + "...")

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                st.markdown(
                    f"[👉 Xem kênh](https://www.youtube.com/channel/{channel_id})"
                )

            with col_btn2:
                if st.button("📊 Phân tích", key=f"analyze_{channel_id}"):
                    st.session_state.selected_channel_id = channel_id
                    st.session_state.selected_channel_name = channel_title
                    st.switch_page("app.py")