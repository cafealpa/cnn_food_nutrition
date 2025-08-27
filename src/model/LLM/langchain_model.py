import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.agents import create_openai_functions_agent, tool, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.logic.food_nutrition_service import get_food_nutrition_info

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 분석용 툴 정의
@tool
def get_food_info(food_name: str) -> str:
    """주어진 음식 이름에 해당하는 영양 정보를 반환합니다."""
    result = get_food_nutrition_info([food_name])
    if result:
        return str(result)
    else:
        return None


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

nutrition_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "역할: 당신은 식품 영양 분석 전문가입니다.\n"
     "목표:\n"
     "1) 제공된 도구(get_food_info)로 입력된 음식의 영양 데이터를 조회합니다.\n"
     "2) 조회된 데이터가 없으면,  일반적인 영양 지식을 기반으로 해당 음식의 성분을 추정합니다.\n"
     "3) 영양 데이터를 근거로 건강 점수(0~100)를 산출하고, 과/부족 항목을 설명합니다.\n"
     "4) 개선 팁(예: 나트륨 낮추기, 단백질 보완)을 제안합니다.\n"
     "규칙:\n"
     "- 데이터가 있으면 반드시 도구를 우선 사용할 것.\n"
     "- 데이터가 없으면 추정하되, 반드시 실제 음식 유형에 맞도록 할 것.\n"
     "- 출력은 마지막에 깔끔한 한국어 문단으로 제공.\n"),
    ("human",
     "{input}\n"
     "출력 형식 가이드:\n"
     "1) 건강 점수: NN/100\n"
     "2) 이유: (성분별 근거)\n"
     "3) 개선 팁: (실천 가능한 제안 2~4개)"),
    MessagesPlaceholder("agent_scratchpad"),
])


agent = create_openai_functions_agent(llm, tools=[get_food_info], prompt=nutrition_prompt)
agent_executor = AgentExecutor(agent=agent, tools=[get_food_info], verbose=True)


# 실행 예시
if __name__ == "__main__":
    food_name = "과메기"
    query = f"""
    {food_name}의 영양 정보를 바탕으로 
    1) 건강 점수를 0~100으로 매겨줘.
    2) 점수 이유를 성분별로 설명해줘.
    3) 개선 팁을 알려줘.
    """

    response = agent_executor.invoke({"input": query})
    print(response["output"])