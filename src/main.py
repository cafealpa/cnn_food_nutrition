import streamlit as st
from service.img import get_image_from_uploader
from service.predict import predict
from service.food_nutrition_service import ask_llm_for_ui
from streamlit_star_rating import st_star_rating
import pandas as pd

def arranged_text(raw_text):
    """텍스트를 정리하여 HTML에서 사용할 수 있도록 포맷팅합니다."""
    text_group = ""
    for raw in raw_text:
        raw1 = raw.replace("-**", "")
        raw2 = raw1.replace("**", "")
        # 줄바꿈을 <br> 태그로 변환하여 HTML에서 제대로 렌더링
        text_group += raw2.strip() + "<br>"
    
    # 마지막 <br> 태그 제거
    text_group = text_group.rstrip("<br>")
    return text_group

def one_line_text(raw_text):
    text_group = ""
    raw1 = raw_text.replace("-**", "")
    raw2 = raw1.replace("**", "")
    # 줄바꿈을 <br> 태그로 변환하여 HTML에서 제대로 렌더링
    text_group += raw2.strip() + "<br>"
    text_group = text_group.rstrip("<br>")

    return text_group


@st.fragment
def result_fragment():
    if st.session_state.current_score is not None:
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
                
                if st.session_state.image_classified_or_not == True:
                    # 점수
                    with st.container(border=False, gap=None):
                        star_value = int(float(st.session_state.current_score / 100) * 5)
                        print("scoring")
                        print(st.session_state.current_score)
                        print(star_value)
                        st_star_rating("", read_only=True, maxValue=5, defaultValue=star_value, key="rating_widget")
                    # 영양 성분
                    with st.container(border=False, gap=None):
                        nuts = st.session_state.current_nutrients
                        # print(nuts)
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "칼로리(kCal)": [nuts["열량(kcal)"] if nuts["열량(kcal)"] != None else 0 ],
                                    "탄수화물(g)": [nuts["탄수화물(g)"] if nuts["탄수화물(g)"] != None else 0 ],
                                    "단백질(g)": [nuts["단백질(g)"] if nuts["단백질(g)"] != None else 0 ],
                                    "지방(g)": [nuts["지방(g)"] if nuts["지방(g)"] != None else 0 ],
                                    "당(g)": [nuts["당(g)"] if nuts["당(g)"] != None else 0 ],
                                }
                            ),
                            hide_index=True,
                        )
                else:
                    with st.container(border=False, gap=None):
                        st.write("분류할수 없는 음식입니다")
            if st.session_state.image_classified_or_not == True:
                st.markdown("------")

                style_sheet = """
                            <style>
                                .title {
                                    font-size: 24px;
                                    font-weight: bold;
                                    margin: 0;
                                    padding: 0;
                                }
                                .contents {
                                    font-size: 16px;
                                    margin: 4px 0 12px 0;
                                    padding: 0;
                                    line-height: 1.4;
                                }
                                .container {
                                    margin: 0;
                                    padding: 0;
                                }
                                .score-text, .score-reason, .score-tips {
                                    margin: 0;
                                    padding: 0;
                                }
                                .mid-title {
                                    margin-top: 16px;
                                }
                                .one-line-result {
                                    margin-top: 24px;
                                    font-weight: bold;
                                    font-size: 20px;
                                }
                            </style>
                """
                # GPT 내용
                with st.container():
                    st.html(style_sheet + f"""
                            <div style="border-radius: 8px; background-color: rgba(127, 127, 127, 0.5); padding: 16px; margin: 0;">
                                <div class="container" style="line-height: 64px;">
                                    <div class="title">💯 건강점수</div>
                                    <div class="contents score-text">{st.session_state.score_text}</div>
                                    <div class="title mid-title">📊 영양소별 분석</div>
                                    <div class="contents score-reason">{arranged_text(st.session_state.reason)}</div>
                                    <div class="title mid-title">👍 식생활 개선 팁</div>
                                    <div class="contents score-tips">{arranged_text(st.session_state.tips[:-1])}</div>
                                    <div class="contents one-line-result">{one_line_text(st.session_state.tips[-1])}</div>
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
    if "current_score" not in st.session_state:
        st.session_state.current_score = None
    if "current_nutrients" not in st.session_state:
        st.session_state.current_nutrients = None
    if "score_text" not in st.session_state:
        st.session_state.score_text = None
    if "reason" not in st.session_state:
        st.session_state.reason = None
    if "tips" not in st.session_state:
        st.session_state.tips = None
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "current_image_name" not in st.session_state:
        st.session_state.current_image_name = None
    if "current_image_confidence" not in st.session_state:
        st.session_state.current_image_confidence = None
    if "image_classified_or_not" not in st.session_state:
        st.session_state.image_classified_or_not = None

    if uploaded_file is not None:
        if st.session_state.current_file_name != uploaded_file.name:
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.current_score = None
            st.session_state.current_nutrients = None
            st.session_state.score_text = None
            st.session_state.reason = None
            st.session_state.tips = None
            st.session_state.current_image = None
            st.session_state.current_image_name = None
            st.session_state.current_image_confidence = None
            st.session_state.image_classified_or_not = None
            # 이전 결과 관련 세션 상태 초기화
            for key in list(st.session_state.keys()):
                if key not in ["current_file_name", "food_image_uploader", "current_score", "image_classified_or_not"]:
                    del st.session_state[key]
            st.rerun()  # 전체 앱 재실행으로 변경

        # 이미지 처리 및 예측 (새 파일일 때만)
        if st.session_state.current_score is None:
            st.session_state.current_image = uploaded_file
            img_array = get_image_from_uploader(uploaded_file)
            # 예측 코드
            # 첫번째는 이미지 배열, 두번째는 모델 경로, 세번째는 class_indices경로를 넣어주면 됩니다!
            pred = predict(img_array, 
                           "model/models/cho_korean_food_classifier-fine-20250827-161229.keras",
                           "model/models/indices-fine-20250827-161229.json")
            
            st.session_state.current_image_confidence = pred['confidence']
            if float(pred['confidence']) < 40.0:
                st.session_state.image_classified_or_not = False
                st.session_state.current_image_name = "미분류"
                st.session_state.current_score = 1
            else:
                st.session_state.image_classified_or_not = True
                st.session_state.current_image_name = pred['predict'][0]
                # LLM 호출 코드
                results = ask_llm_for_ui(pred['predict'][0])
                st.session_state.current_score = results["score"]
                st.session_state.current_nutrients = results["nutrients"]
                st.session_state.score_text = results["analysis"]["score_text"]
                st.session_state.reason = results["analysis"]["reason"]
                st.session_state.tips = results["analysis"]["tips"]

        # 결과 컨테이너 - fragment로 독립적으로 렌더링
        result_fragment()

if __name__ == "__main__":
    st.set_page_config(
        page_title="AI 음식 검사", 
        page_icon="🥣",
        layout="wide"
    )
    main()

    