import sys
import os
sys.path.append(r"C:\Users\senbo\Desktop\taba_project\ai_series\coai_s.py")
import ledger_s
import coai_s
import test_s_1

def start_ledger():
    image_path = r"C:\Users\senbo\Desktop\python\rrr\test\ledger.jpg"
    output_file_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\ledger_result.json"
    ledger_s.request(image_path, output_file_path)

def start_coai():
    img_list = [rf"C:\Users\senbo\Desktop\python\rrr\test_st\con_{i}.jpg" for i in range(1,4)]
    # output_path_2=r"C:\Users\senbo\Desktop\taba\python\rrr\test\co_m.jpg"
    output_file_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\coai_result_a.json"
    coai_s.request(img_list,output_file_path)

def start_reg():
    img_list = [rf"C:\Users\senbo\Desktop\python\rrr\test\000{i}.jpg" for i in range(1,4)]
    output_file_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\reg_result.json"
    output_file_path_2 = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\reg_m.jpg"
    test_s_1.request(img_list, output_file_path, output_file_path_2)

start_ledger()
