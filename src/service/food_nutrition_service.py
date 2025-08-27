# 필요한 라이브러리들을 가져옵니다.
import sys
import os
import re
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# import src.db.database as DB
from src.db import database as DB
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.agents import create_openai_functions_agent, tool, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# .env 파일에서 환경 변수를 불러옵니다 (API 키 등)
load_dotenv()
# OpenAI 클라이언트를 초기화합니다.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# LangChain에서 사용할 언어 모델(LLM)을 설정합니다. 여기서는 gpt-4o-mini를 사용합니다.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LLM 에이전트에게 전달할 시스템 프롬프트입니다.
# 에이전트의 역할, 목표, 규칙을 정의하여 행동을 제어합니다.
system_prompt = """
    역할: 당신은 식품 영양 분석 전문가입니다.
    목표:
    1) 제공된 도구(get_food_info)로 입력된 음식의 영양 데이터를 조회합니다.
    2) 조회된 데이터가 없으면,  일반적인 영양 지식을 기반으로 해당 음식의 성분을 추정합니다.
    3) 영양 데이터를 근거로 건강 점수(0~100)를 산출하고, 과/부족 항목을 설명합니다.
    4) 개선 팁(예: 나트륨 낮추기, 단백질 보완)을 제안합니다.
    규칙:
    - 데이터가 있으면 반드시 도구를 우선 사용할 것.
    - 데이터가 없으면 추정하되, 반드시 실제 음식 유형에 맞도록 할 것.
    - 출력은 마지막에 깔끔한 한국어 문단으로 제공.,
"""

# 사용자의 입력을 처리하기 위한 프롬프트 템플릿입니다.
# 출력 형식을 지정하여 일관된 결과를 얻도록 유도합니다.
human_prompt = """
    {input}
    출력 형식 가이드:
    1) 건강 점수: NN/100
    2) 이유: (성분별 근거)
    3) 개선 팁: (실천 가능한 제안 2~4개)"),
"""


def get_food_nutrition_info(food_names: list):
    """
    주어진 음식 이름 목록을 기반으로 데이터베이스에서 음식 영양 정보를 조회합니다.
    '외식'으로 분류된 음식을 우선적으로 반환하며, 없을 경우 첫 번째 검색 결과를 반환합니다.

    Args:
        food_names (list): 조회할 음식 이름의 리스트.

    Returns:
        dict: 조회된 음식의 영양 정보 데이터 (딕셔너리).
              데이터를 찾지 못한 경우 None을 반환합니다.
    """
    for food_name in food_names:
        # 데이터베이스 모듈을 통해 음식 정보를 가져옵니다.
        food_datas = DB.get_food_info_by_name(food_name)

        if not food_datas:
            return None
        else:
            # '외식' 데이터가 있으면 우선적으로 반환합니다.
            for food_data in food_datas:
                if '외식 ' in food_data.get("food_origin_name", ""):
                    return food_data
            # '외식' 데이터가 없으면 첫 번째 결과를 반환합니다.
            return food_datas[0]
    return None


# @tool 데코레이터는 이 함수를 LangChain 에이전트가 사용할 수 있는 도구로 만들어줍니다.
@tool
def get_food_info(food_name: str) -> str:
    """주어진 음식 이름에 해당하는 영양 정보를 문자열로 반환하는 도구입니다."""
    result = get_food_nutrition_info([food_name])
    if result:
        return str(result)
    else:
        return "데이터베이스에서 해당 음식 정보를 찾을 수 없습니다."


def ask_llm(food_name):
    """
    LLM 에이전트를 설정하고 실행하여 음식에 대한 영양 분석을 요청합니다.

    Args:
        food_name (str): 분석할 음식의 이름.

    Returns:
        str: LLM 에이전트가 생성한 최종 분석 결과.
    """
    # 에이전트가 사용할 프롬프트 템플릿을 구성합니다.
    nutrition_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
        MessagesPlaceholder("agent_scratchpad"),  # 에이전트의 중간 작업 과정을 저장하는 공간
    ])

    # LLM, 도구, 프롬프트를 연결하여 에이전트를 생성합니다.
    agent = create_openai_functions_agent(llm, tools=[get_food_info], prompt=nutrition_prompt)
    # 생성된 에이전트를 실행할 실행기(Executor)를 만듭니다.
    agent_executor = AgentExecutor(agent=agent, tools=[get_food_info], verbose=True)

    # 사용자 입력을 받아 프롬프트 형식에 맞게 구성합니다.
    input_prompt = f"""
        {food_name}의 영양 정보를 바탕으로 
        1) 건강 점수를 0~100으로 매겨줘.
        2) 점수 이유를 성분별로 설명해줘.
        3) 개선 팁을 알려줘.
    """
    # 에이전트 실행기에게 입력을 전달하여 결과를 얻습니다.
    response = agent_executor.invoke({"input": input_prompt})
    return response["output"]



def ask_llm_for_ui(food_name: str):
    """
    UI에 바로 사용할 수 있는 영양 분석 결과를 반환합니다.
    반환 형식(딕셔너리):
    {
        "score": int,  # 건강 점수 (0~100)
        "nutrients": dict,  # 영양소 데이터
        "analysis": {
            "score_text": str,   # 예: "75/100"
            "reason": [str, ...],  # 이유 
            "tips": [str, ...]     # 개선 팁 
        }
    }
    """

    response_text = ask_llm(food_name)

    # 1) 건강 점수
    score_match = re.search(r"건강\s*점수[:\s]+(\d+)", response_text)
    score = int(score_match.group(1)) if score_match else 0
    score_text = f"{score}/100"

    # 2) 영양소 데이터 (DB에서 가져오기)
    food_data = get_food_nutrition_info([food_name])
    nutrients = {
        "열량(kcal)": food_data.get("energy_kcal", 0) if food_data else 0,
        "탄수화물(g)": food_data.get("carbohydrates_g", 0) if food_data else 0,
        "단백질(g)": food_data.get("protein_g", 0) if food_data else 0,
        "지방(g)": food_data.get("fat_g", 0) if food_data else 0,
        "당(g)": food_data.get("sugars_g", 0) if food_data else 0,
    }

    # 3) 이유
    reason_match = re.search(r"2\)\s*이유[:\s]*([\s\S]*?)(?:3\)|$)", response_text)
    reasons = []
    if reason_match:
        reason_lines = reason_match.group(1).strip().split("\n")
        reasons = [line.lstrip("-• ").strip() for line in reason_lines if line.strip()]

    # 4) 개선 팁
    tips_match = re.search(r"3\)\s*개선\s*팁[:\s]*([\s\S]*)", response_text)
    tips = []
    if tips_match:
        tip_lines = tips_match.group(1).strip().split("\n")
        tips = [line.lstrip("-• ").strip() for line in tip_lines if line.strip()]


    return {
        "score": score,
        "nutrients": nutrients,
        "analysis": {
            "score_text": score_text,
            "reason": reasons,
            "tips": tips
        }
    }



# 이 스크립트가 직접 실행될 때만 아래 코드를 실행합니다. (테스트용)
if __name__ == "__main__":
    # '김치찜'에 대한 영양 분석을 요청하고 결과를 출력합니다.
    print(ask_llm("김치찜"))
