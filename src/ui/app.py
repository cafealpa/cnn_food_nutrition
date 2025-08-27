import streamlit as st
from service.img import get_image_from_uploader
from service.predict import predict
from service.api import connection_api
from streamlit_star_rating import st_star_rating
import pandas as pd

db_converted = {
    "간장게장": "게장_간장",
    "감자채볶음": "감자볶음",
    "계란국": "달걀국",
    "계란말이": "달걀말이",
    "계란찜": "달걀찜",
    "계란후라이": "달걀후라이",
    "고추장진미채볶음": "오징어볶음",
    "곰탕_설렁탕": "설렁탕",
    "꽈리고추무침": "오이지무침_고추",
    "닭계장": "닭볶음탕",
    "도라지무침": "도라지생채",
    "떡국_만두국": "떡국_소고기",
    "떡꼬치": "떡강정",
    "북엇국": "북어국",
    "새우볶음밥": "볶음밥_새우",
    "소세지볶음": "소시지케첩볶음",
    "시래기국": "된장국_시래기",
    "양념게장": "게장_양념",
    "열무국수": "국수_열무김치",
    "젓갈": "양념오징어젓",
    "편육": "수육",
    "한과": "유과",
    "과메기": None,
    "산낙지": None,
    "수정과": None
}

@st.fragment
def result_fragment():
    if st.session_state.current_result is not None:
        with st.container(border=True):
            img_col, summary_col = st.columns([1, 3])
            with img_col:
                st.image(st.session_state.current_image)
            with summary_col:
                title, conf = st.columns([1, 1], vertical_alignment="center", gap=None)
                with title:
                    with st.container(border=False, width="stretch", height="stretch", vertical_alignment="bottom", horizontal_alignment="left", gap=None):
                        st.html(f"<p style='margin-bottom: 0;'><span style='font-size: 36px; font-weight: bold;'>{st.session_state.current_image_name}</span></p>")
                with conf:
                    badge_color = 'red'
                    if float(st.session_state.current_image_confidence) > 50:
                        badge_color = 'green'
                    else:
                        badge_color = 'red'
                    with st.container(border=False, width="stretch", height="stretch", vertical_alignment="bottom", horizontal_alignment="right", gap=None):
                        st.badge(f"{st.session_state.current_image_confidence}%", icon="💯", color=badge_color, width="content")
                # 점수
                with st.container(border=False, gap=None):
                    st_star_rating("", read_only=True, maxValue=5, defaultValue=3, key="rating_widget")
                # 영양 성분
                with st.container(border=False, gap=None):
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "칼로리(kCal)": [1],
                                "탄수화물(g)": [10],
                                "단백질(g)": [10],
                                "지방(g)": [10],
                                "당(g)": [10],
                            }
                        ),
                        hide_index=True,
                    )
            st.markdown("------")
            # GPT 내용
            with st.container():
                st.html("""
                        <div style="border-radius: 8px; background-color: rgba(127, 127, 127, 0.5); padding: 8px;">
                            <div class="contents">
                                이것은 테스트 내용입니다
                            </div>
                        </div>
                        """)


def main():
    st.title("🥣 AI 음식 검사")
    st.badge("음식 사진을 업로드해서 좋은 음식인지 나쁜 음식인지 알아보세요", color="blue")
    st.divider()

    uploaded_file = st.file_uploader("아래 버튼을 눌러 사진을 업로드해주세요", type=["jpg", "jpeg", "png"], accept_multiple_files=False, label_visibility="visible", width="stretch", key="food_image_uploader")
    
    # 현재 업로드된 파일명 추적
    if "current_file_name" not in st.session_state:
        st.session_state.current_file_name = None
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_image_name" not in st.session_state:
        st.session_state.current_image_name = None
    if "current_image_confidence" not in st.session_state:
        st.session_state.current_image_confidence = None

    if uploaded_file is not None:
        if st.session_state.current_file_name != uploaded_file.name:
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.current_result = None  # 결과 초기화
            st.session_state.current_image = None
            st.session_state.current_image_name = None
            st.session_state.current_image_confidence = None
            # 이전 결과 관련 세션 상태 초기화
            for key in list(st.session_state.keys()):
                if key not in ["current_file_name", "food_image_uploader", "current_result"]:
                    del st.session_state[key]
            st.rerun()  # 전체 앱 재실행으로 변경

        # 이미지 처리 및 예측 (새 파일일 때만)
        if st.session_state.current_result is None:
            st.session_state.current_image = uploaded_file
            img_array = get_image_from_uploader(uploaded_file)
            pred = predict(img_array)
            st.session_state.current_image_name = pred['predict'][0]
            st.session_state.current_image_confidence = pred['confidence']
            result = connection_api(pred, [])
            st.session_state.current_result = result

        # 결과 컨테이너 - fragment로 독립적으로 렌더링
        result_fragment()

if __name__ == "__main__":
    st.set_page_config(
        page_title="AI 음식 검사", 
        page_icon="🥣",
        layout="wide"
    )
    main()

    