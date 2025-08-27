#  음식 이미지 기반 영양 정보 분석 AI

이 프로젝트는 음식 사진을 분석하여 해당 음식의 종류를 식별하고, 영양 정보를 제공하며, 관련된 추가 정보를 생성하는 딥러닝 및 LLM 기반 애플리케이션입니다.

사용자는 음식 사진을 업로드하기만 하면, CNN 모델이 음식을 인식하고, LangChain과 OpenAI API를 통해 상세한 영양 정보와 설명을 얻을 수 있습니다.

## 📝 개요

음식 사진 한 장으로 간편하게 영양 정보를 확인하고 건강 관리에 도움을 주는 것을 목표로 합니다. TensorFlow로 학습된 CNN 모델을 통해 이미지 속 음식을 분류하고, 분류된 결과를 바탕으로 Pandas로 관리되는 데이터에서 영양 정보를 조회합니다. 마지막으로, LangChain과 OpenAI의 LLM을 활용하여 사용자에게 보다 풍부하고 이해하기 쉬운 형태로 정보를 가공하여 제공합니다.

## ✨ 주요 기능

- **음식 이미지 분류**: TensorFlow/Keras 기반의 CNN(Convolutional Neural Network) 모델을 사용하여 이미지 속 음식을 정확하게 식별합니다.
- **영양 정보 제공**: 식별된 음식을 기반으로 칼로리, 단백질, 탄수화물, 지방 등 주요 영양 정보를 제공합니다. (Pandas를 활용하여 데이터 관리)
- **LLM을 통한 상세 정보 생성**: OpenAI (GPT) 모델을 활용하여 음식에 대한 설명, 건강 팁, 또는 간단한 레시피 등 풍부한 정보를 동적으로 생성합니다.

## ⚙️ 시스템 동작 원리

1.  **사용자 입력**: 사용자가 음식 이미지를 시스템에 입력합니다.
2.  **이미지 처리 및 예측**: `OpenCV`로 이미지를 전처리하고, 학습된 `TensorFlow` CNN 모델이 이미지를 분석하여 음식 종류를 예측합니다.
3.  **정보 조회 및 생성**:
    - 예측된 음식 이름을 기반으로 `Pandas` 데이터프레임에서 기본 영양 정보를 조회합니다.
    - `LangChain`과 `OpenAI` API를 호출하여 조회된 정보를 바탕으로 사용자 친화적인 설명과 추가 정보를 생성합니다.
4.  **결과 출력**: 최종적으로 분석된 영양 정보와 LLM이 생성한 상세 설명을 사용자에게 보여줍니다.

## 🚀 설치 방법

프로젝트를 로컬 환경에서 실행하기 위한 절차는 다음과 같습니다.

1.  **리포지토리 클론:**

    ```bash
    git clone <your-repository-url>
    cd cnn_food_nutrition
    ```

2.  **가상 환경 생성 및 활성화:**

    프로젝트 의존성 관리를 위해 가상 환경 사용을 권장합니다.

    ```bash
    # Python 가상 환경 생성
    python -m venv venv

    # Windows에서 활성화
    .\venv\Scripts\activate

    # macOS/Linux에서 활성화
    source venv/bin/activate
    ```

3.  **의존성 패키지 설치:**

    `requirements.txt` 파일에 명시된 모든 패키지를 설치합니다.

    ```bash
    pip install -r requirements.txt
    ```

4.  **환경 변수 설정:**

    프로젝트 루트 디렉토리에 `.env` 파일을 생성하고, 파일 내에 OpenAI API 키를 추가합니다.

    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## ▶️ 사용 방법

1.  프로젝트의 메인 스크립트를 실행하여 프로그램을 시작합니다. (예: `main.py`)
    ```bash
    python main.py
    ```
2.  프로그램의 안내에 따라 분석하고 싶은 음식 이미지의 경로를 입력하거나 파일을 선택합니다.
3.  분석 결과를 터미널 또는 GUI 화면에서 확인합니다.

## 🛠 기술 스택

- **Deep Learning**: `TensorFlow`
- **Image Processing**: `OpenCV-Python`
- **Numerical Computing**: `NumPy`
- **Data Handling**: `Pandas`
- **LLM Framework**: `LangChain`, `LangChain-OpenAI`
- **LLM API**: `OpenAI`
- **Environment Variables**: `Python-Dotenv`