
"""
1. 타겟 폴더의 경로를 입력받고
2. 타겟 폴더 가장 하위의 폴더들로 접근해서
3. 인자로 입력받은 갯수 만큼 파일을 읽어
4. 하위의 폴더들에는 'crop_area.properties'파일이 있는 경우는 프로퍼티 파일을 읽어서 해당파일명이 있으면 프로퍼티에 입력되어 있는 좌표대로 이미지를 크롭한다.
5. 크롭된 이미지를 인자로 입력된 dest폴더로 복사한다  
6. 'crop_area.properties'파일이 없으면 dest폴더로 복사한다. 
"""

import os
import shutil
import cv2
import numpy as np
from typing import List, Dict, Tuple
import time
import json
import random


def read_properties(prop_file: str) -> Dict[str, Tuple[int, int, int, int]]:
    """
    프로퍼티 파일을 읽어 이미지 파일별 크롭 영역 좌표를 반환합니다.

    Args:
        prop_file (str): 'crop_area.properties' 파일의 경로

    Returns:
        Dict[str, Tuple[int, int, int, int]]: 파일명을 키로, (x, y, w, h) 좌표를 값으로 하는 딕셔너리
    """
    crop_areas = {}
    if os.path.exists(prop_file):
        with open(prop_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    name, coords = line.strip().split('=', 1)
                    coords = coords.split()[0]

                    if coords.startswith(',') or coords.startswith(('dissimilar', 'open')):
                        split_coords = coords.split(',')
                        x = 0
                        y = 0
                        w = int(split_coords[1])
                        h = int(split_coords[2])
                    else:
                        split_coords = coords.split(',')
                        x = int(split_coords[0])
                        y = int(split_coords[1])
                        w = int(split_coords[2])
                        h = int(split_coords[3])
                    crop_areas[name] = (x, y, w, h)
    return crop_areas


def get_lowest_dirs(target_dir: str) -> List[str]:
    """
    주어진 디렉토리의 모든 최하위 디렉토리 목록을 반환합니다.

    Args:
        target_dir (str): 대상 디렉토리 경로

    Returns:
        List[str]: 최하위 디렉토리 경로의 리스트
    """
    lowest_dirs = []
    for root, dirs, files in os.walk(target_dir):
        if not dirs:
            lowest_dirs.append(root)
    return lowest_dirs


def crop_image(image_path: str, coords: Tuple[int, int, int, int]) -> any:
    """
    주어진 좌표에 따라 이미지를 자릅니다. 한글 경로 문제를 해결하기 위해 numpy로 파일을 읽습니다.

    Args:
        image_path (str): 이미지 파일 경로
        coords (Tuple[int, int, int, int]): (x, y, 너비, 높이) 형식의 자르기 좌표

    Returns:
        any: 잘린 이미지 객체 (OpenCV 이미지). 실패 시 None을 반환합니다.
    """
    try:
        with open(image_path, 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError("Failed to decode image")
    except Exception as e:
        print(f"Error: 이미지 읽기 실패 {image_path} - {e}")
        return None
    
    x, y, w, h = coords
    return img[y:y + h, x:x + w]


def train_files_pre_process(target_dir, dest_dir, count, is_valid=False):
    """
    대상 디렉토리의 이미지 파일들을 전처리하여 목적 디렉토리로 복사합니다.

    'crop_area.properties' 파일이 있는 경우, 해당 파일의 좌표 정보를 이용해 이미지를 자른 후 복사합니다.
    파일이 없으면 원본 이미지를 그대로 복사합니다.
    한글 경로 문제를 해결하기 위해 cv2.imencode를 사용하여 파일을 저장합니다.

    Args:
        target_dir (str): 원본 이미지 파일들이 있는 대상 디렉토리
        dest_dir (str): 전처리된 파일들을 저장할 목적 디렉토리
        count (int): 각 하위 폴더에서 처리할 파일의 수. 0이면 모든 파일을 처리합니다.

    Returns:
        dict: 처리된 파일명을 키로, {'folder': 원본폴더명, 'type': 파일유형}을 값으로 하는 딕셔너리
    """

    if is_valid:
        count = int(count * 0.3)


    os.makedirs(dest_dir, exist_ok=True)
    result = {}

    for dir_path in get_lowest_dirs(target_dir):
        # Use only the last directory name for destination
        folder_type = os.path.basename(dir_path)
        current_dest_dir = os.path.join(dest_dir, folder_type)
        os.makedirs(current_dest_dir, exist_ok=True)

        prop_file = os.path.join(dir_path, 'crop_area.properties')
        crop_areas = read_properties(prop_file)

        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
                 and f != 'crop_area.properties']

        if count > 0:
            # Set random seed based on current time
            random.seed(int(time.time()))
            files = random.sample(files, min(count, len(files)))

        for file in files:
            src_path = os.path.join(dir_path, file)
            dest_path = os.path.join(current_dest_dir, file)

            # 파일명이 중복되면 폴더명을 포함하여 이름 변경
            base, ext = os.path.splitext(file)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(current_dest_dir, f"{base}_{folder_type}_{counter}{ext}")
                counter += 1

            try:
                file_name_only = os.path.splitext(file)[0]
                if file_name_only in crop_areas:
                    cropped = crop_image(src_path, crop_areas[file_name_only])
                    if cropped is not None:
                        extension = os.path.splitext(dest_path)[1]
                        result_encode, encoded_img = cv2.imencode(extension, cropped)
                        if result_encode:
                            with open(dest_path, 'wb') as f:
                                f.write(encoded_img)
                            result[file] = folder_type
                        else:
                            print(f"Warning: 이미지 인코딩 실패 {src_path}")
                    else:
                        print(f"Warning: 이미지 크롭 실패 {src_path}")
                else:
                    for attempt in range(3):
                        try:
                            shutil.copy2(src_path, dest_path)
                            result[file] = {'folder': dir_path, 'type': folder_type}
                            break
                        except PermissionError as e:
                            print(f"파일 사용 중, {file} - {e}, 재시도 {attempt + 1}/3")
                            time.sleep(1)
                    else:
                        print(f"복사 실패: {file}")
            except Exception as e:
                print(f"처리 중 오류 발생 {file}: {e}")

    return result


if __name__ == '__main__':
    target_dir = "E:/AIWork/Data/한국음식"
    train_dest_dir = "E:/AIWork/Data/테스트/train"
    valid_dest_dir = "E:/AIWork/Data/테스트/valid"
    count = 100

    # for dest_dir in [train_dest_dir, valid_dest_dir]:
    #     result = train_files_pre_process(target_dir, dest_dir, count)
    #     # Save process_result to JSON file
    #     result_file = os.path.join(dest_dir, 'result.json')
    #     with open(result_file, 'w', encoding='utf-8') as f:
    #         json.dump(dest_dir, f, ensure_ascii=False, indent=2)

    # result = train_files_pre_process(target_dir, train_dest_dir, count)
    # # Save process_result to JSON file
    # result_file = os.path.join(train_dest_dir, 'result.json')
    # with open(result_file, 'w', encoding='utf-8') as f:
    #     json.dump(train_dest_dir, f, ensure_ascii=False, indent=2)

    result_valid = train_files_pre_process(target_dir, valid_dest_dir, count, is_valid=True)
    # Save process_result to JSON file
    result_valid_file = os.path.join(valid_dest_dir, 'result.json')
    with open(result_valid_file, 'w', encoding='utf-8') as f:
        json.dump(valid_dest_dir, f, ensure_ascii=False, indent=2)


    print("종료")
