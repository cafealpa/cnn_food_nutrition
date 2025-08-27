import csv
import sqlite3

def migrate_data():
    # CSV 파일 경로와 데이터베이스 경로
    csv_file_path = '../../data.csv'
    db_file_path = '../../food_nutrition.db'

    # 한글 헤더와 영문 컬럼명 매핑
    header_mapping = {
        '식품코드': 'food_code',
        '식품명': 'food_name',
        '데이터구분코드': 'data_division_code',
        '데이터구분명': 'data_division_name',
        '식품기원코드': 'food_origin_code',
        '식품기원명': 'food_origin_name',
        '식품대분류코드': 'food_main_category_code',
        '식품대분류명': 'food_main_category_name',
        '대표식품코드': 'representative_food_code',
        '대표식품명': 'representative_food_name',
        '식품중분류코드': 'food_mid_category_code',
        '식품중분류명': 'food_mid_category_name',
        '식품소분류코드': 'food_sub_category_code',
        '식품소분류명': 'food_sub_category_name',
        '식품세분류코드': 'food_detail_category_code',
        '식품세분류명': 'food_detail_category_name',
        '영양성분함량기준량': 'nutrition_content_standard_amount',
        '에너지(kcal)': 'energy_kcal',
        '수분(g)': 'moisture_g',
        '단백질(g)': 'protein_g',
        '지방(g)': 'fat_g',
        '회분(g)': 'ash_g',
        '탄수화물(g)': 'carbohydrates_g',
        '당류(g)': 'sugars_g',
        '식이섬유(g)': 'dietary_fiber_g',
        '칼슘(mg)': 'calcium_mg',
        '철(mg)': 'iron_mg',
        '인(mg)': 'phosphorus_mg',
        '칼륨(mg)': 'potassium_mg',
        '나트륨(mg)': 'sodium_mg',
        '비타민 A(μg RAE)': 'vitamin_a_ug_rae',
        '레티놀(μg)': 'retinol_ug',
        '베타카로틴(μg)': 'beta_carotene_ug',
        '티아민(mg)': 'thiamine_mg',
        '리보플라빈(mg)': 'riboflavin_mg',
        '니아신(mg)': 'niacin_mg',
        '비타민 C(mg)': 'vitamin_c_mg',
        '비타민 D(μg)': 'vitamin_d_ug',
        '콜레스테롤(mg)': 'cholesterol_mg',
        '포화지방산(g)': 'saturated_fatty_acids_g',
        '트랜스지방산(g)': 'trans_fatty_acids_g',
        '출처코드': 'source_code',
        '출처명': 'source_name',
        '식품중량': 'food_weight',
        '전체내용량': 'total_content',
        '데이터생성방법코드': 'data_creation_method_code',
        '데이터생성방법명': 'data_creation_method_name',
        '데이터생성일자': 'data_creation_date',
        '데이터기준일자': 'data_reference_date',
        '제공기관코드': 'provider_code',
        '제공기관명': 'provider_name'
    }

    # 데이터베이스 연결
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    # CSV 파일 열기
    with open(csv_file_path, 'r', encoding='cp949') as f:
        reader = csv.reader(f)
        header = next(reader)
        cleaned_header = [h.strip().lstrip('\ufeff') for h in header]

        # 헤더를 영문명으로 변환
        english_columns = []
        original_header_for_insert = []
        for h in cleaned_header:
            mapped_col = header_mapping.get(h)
            if mapped_col:
                english_columns.append(mapped_col)
                original_header_for_insert.append(h)

        # 유효한 헤더가 있는지 확인
        if not english_columns:
            print("오류: 유효한 헤더를 찾을 수 없습니다.")
            conn.close()
            return

        # 테이블 생성 (기존 테이블이 있으면 삭제 후 다시 생성)
        cursor.execute("DROP TABLE IF EXISTS food_nutrition")
        columns_with_types = [f'"{col_name}" TEXT' for col_name in english_columns]
        create_table_sql = f"CREATE TABLE food_nutrition ({ ', '.join(columns_with_types) })"
        cursor.execute(create_table_sql)

        # 데이터 삽입
        header_indices = [i for i, h in enumerate(cleaned_header) if h in original_header_for_insert]
        
        placeholders = ', '.join(['?' for _ in english_columns])
        insert_sql = f'INSERT INTO food_nutrition ("{ '", "'.join(english_columns) }") VALUES ({placeholders})'

        for row in reader:
            selected_row_data = [row[i] for i in header_indices if i < len(row)]
            if len(selected_row_data) == len(english_columns):
                cursor.execute(insert_sql, selected_row_data)
            else:
                print(f"컬럼 수가 일치하지 않아 다음 행을 건너뜁니다: {row}")

    # 변경사항 저장 및 연결 종료
    conn.commit()
    conn.close()

    print(f"'{db_file_path}'에 데이터 마이그레이션을 완료했습니다.")

if __name__ == "__main__":
    migrate_data()