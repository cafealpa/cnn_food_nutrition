import sqlite3
from contextlib import contextmanager

DB_PATH = '../../food_nutrition.db'

@contextmanager
def get_db_connection():
    """
    데이터베이스 연결을 위한 컨텍스트 관리자.
    연결 및 종료를 자동으로 처리합니다.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # 컬럼명으로 접근 가능하도록 설정
        yield conn
    finally:
        if conn:
            conn.close()

def get_food_info_by_name(food_name):
    """
    음식 이름으로 영양 정보를 조회합니다.
    :param food_name: 검색할 음식 이름
    :return: 음식 영양 데이터 리스트
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # SQL Injection을 방지하기 위해 파라미터화된 쿼리(placeholder '?')를 사용합니다.
            # 사용자 입력은 두 번째 인자로 안전하게 전달됩니다.
            query = """
                    SELECT * 
                    FROM FOOD_NUTRITION 
                    WHERE REPLACE(FOOD_NAME, ' ', '') LIKE ?
                    ORDER BY
                        CASE
                            WHEN REPLACE(FOOD_NAME, ' ', '') LIKE ? THEN 1
                            WHEN REPLACE(FOOD_NAME, ' ', '') LIKE ? THEN 2
                            ELSE 3
                        END
                    """
            param = (f'%{food_name}%', f'{food_name}%', f'%{food_name}%')
            cursor.execute(query, param)

            rows = cursor.fetchall()

            return [dict(row) for row in rows]
    except sqlite3.Error as e:
        print(f"데이터베이스 오류: {e}")
        return []
