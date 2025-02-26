import json

def merge_jsons(original_json, analyzed_json):
    """
    두 JSON 파일을 공통 키 기준으로 합치는 함수
    
    Args:
        original_json (dict): 원본 JSON 데이터 (구조 기준)
        analyzed_json (dict): 분석 결과가 포함된 JSON 데이터 (결과 기준)
        
    Returns:
        dict: 합쳐진 JSON 데이터
    """
    # 깊은 복사로 원본 데이터 복제
    merged_json = json.loads(json.dumps(original_json))
    
    # 공통 섹션과 키를 찾아 합치는 함수
    def merge_sections(merged_section, analyzed_section, path=""):
        if not isinstance(merged_section, dict) or not isinstance(analyzed_section, dict):
            return
        
        # 현재 섹션의 모든 키에 대해
        for key in analyzed_section:
            current_path = f"{path}.{key}" if path else key
            
            # 분석 결과에만 있는 키는 건너뛰기
            if key not in merged_section:
                continue
            
            # 두 JSON 모두에 있는 키일 경우
            if isinstance(analyzed_section[key], dict) and isinstance(merged_section[key], dict):
                # "notice"나 "solution" 키가 있으면 복사
                if "notice" in analyzed_section[key]:
                    merged_section[key]["notice"] = analyzed_section[key]["notice"]
                if "solution" in analyzed_section[key]:
                    merged_section[key]["solution"] = analyzed_section[key]["solution"]
                
                # 재귀적으로 하위 객체 병합
                merge_sections(merged_section[key], analyzed_section[key], current_path)
        
    # 주요 섹션(contract, building_registry, registry_document)에 대해 병합
    for section in ["contract", "building_registry", "registry_document"]:
        if section in original_json and section in analyzed_json:
            merge_sections(merged_json[section], analyzed_json[section], section)
    
    return merged_json

def main():
    # 파일에서 JSON 데이터 로드
    with open('sol.json', 'r', encoding='utf-8') as f:
        original_json = json.load(f)
    
    with open('sol_1.json', 'r', encoding='utf-8') as f:
        analyzed_json = json.load(f)
    
    # JSON 병합
    merged_json = merge_jsons(original_json, analyzed_json)
    
    # 결과 저장
    with open('merged_sol.json', 'w', encoding='utf-8') as f:
        json.dump(merged_json, f, ensure_ascii=False, indent=4)
    
    print("두 JSON이 성공적으로 병합되었습니다. 결과는 'merged_sol.json'에 저장되었습니다.")
    
    # 병합 통계 출력
    stats = count_merged_fields(original_json, analyzed_json, merged_json)
    print(f"\n병합 통계:")
    print(f"원본 JSON 필드 수: {stats['original_fields']}")
    print(f"분석 JSON 필드 수: {stats['analyzed_fields']}")
    print(f"병합된 필드 수: {stats['merged_fields']}")
    print(f"notice 필드 추가 수: {stats['notice_added']}")
    print(f"solution 필드 추가 수: {stats['solution_added']}")

def count_merged_fields(original, analyzed, merged):
    """병합 통계 정보를 반환하는 함수"""
    stats = {
        "original_fields": 0,
        "analyzed_fields": 0,
        "merged_fields": 0,
        "notice_added": 0,
        "solution_added": 0
    }
    
    # 필드 수를 세는 함수
    def count_fields(obj, counter_key):
        count = 0
        if isinstance(obj, dict):
            count += 1  # 현재 객체 자체를 카운트
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    count += count_fields(value, counter_key)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    count += count_fields(item, counter_key)
        stats[counter_key] = count
        return count
    
    # notice와 solution 필드 수를 세는 함수
    def count_added_fields(obj):
        notice_count = 0
        solution_count = 0
        
        if isinstance(obj, dict):
            if "notice" in obj:
                notice_count += 1
            if "solution" in obj:
                solution_count += 1
            
            for key, value in obj.items():
                if isinstance(value, dict):
                    n, s = count_added_fields(value)
                    notice_count += n
                    solution_count += s
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            n, s = count_added_fields(item)
                            notice_count += n
                            solution_count += s
        
        return notice_count, solution_count
    
    # 필드 수 계산
    count_fields(original, "original_fields")
    count_fields(analyzed, "analyzed_fields")
    count_fields(merged, "merged_fields")
    
    # notice와 solution 필드 수 계산
    stats["notice_added"], stats["solution_added"] = count_added_fields(merged)
    
    return stats

if __name__ == "__main__":
    main()