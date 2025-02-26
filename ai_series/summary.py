import json

def clean_json(input):
    with open(input, 'r', encoding='utf-8') as f:
        input_json = json.load(f)

    result = {}
    
    # 각 최상위 키에 대해 처리
    for top_key, top_value in input_json.items():
        result[top_key] = {}
        
        # 각 섹션(페이지) 처리
        for section_key, section_value in top_value.items():
            result[top_key][section_key] = {}
            
            # 각 항목 처리
            for item_key, item_value in section_value.items():
                # 딕셔너리가 아닌 경우 건너뛰기
                if not isinstance(item_value, dict):
                    continue
                
                # "notice" 키가 있는 항목만 유지
                if "notice" in item_value:
                    # 새 항목 생성 (bounding_box 제외)
                    new_item = {}
                    for field_key, field_value in item_value.items():
                        if field_key != "bounding_box":
                            new_item[field_key] = field_value
                    
                    # 결과에 추가
                    result[top_key][section_key][item_key] = new_item
            
            # 빈 섹션이면 삭제
            if not result[top_key][section_key]:
                del result[top_key][section_key]
        
        # 빈 최상위 키면 삭제
        if not result[top_key]:
            del result[top_key]
        with open(r'C:\Users\senbo\Desktop\taba_project\ai_series\result\solution_1.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result



clean_json(r'C:\Users\senbo\Desktop\taba_project\ai_series\result\solution.json')




