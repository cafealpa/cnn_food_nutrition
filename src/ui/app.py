import streamlit as st
from streamlit_chat import message

def main():
    st.set_page_config(
        page_title = "음식 영양 분석기",
        page_icon = "🔍",
        layout = "wide"
    )

if __name__ == "__main__":
    main()
