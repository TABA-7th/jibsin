import sys
import os
from dotenv import load_dotenv
load_dotenv(r"C:\Users\senbo\Desktop\taba_project\.env")
sys.path.append(r"C:\Users\senbo\Desktop\taba_project\ai_series")
import ledger_s
import coai_s
import test_s_1

def start_ledger():
    image_path = r"C:\Users\senbo\Desktop\taba_project\test_st\ledger.jpg"
    output_file_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\ledger_result.json"
    ledger_s.request(image_path, output_file_path)
    
def start_coai():
    img_list = [rf"C:\Users\senbo\Desktop\taba_project\test_st\con_{i}.jpg" for i in range(1,4)]
    # output_path_2=r"C:\Users\senbo\Desktop\taba\python\rrr\test\co_m.jpg"
    output_file_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\coai_result_a.json"
    coai_s.request(img_list,output_file_path)

def start_reg():
    img_list = [rf"C:\Users\senbo\Desktop\taba_project\test_sc\reg\000{i}.jpg" for i in range(1,3)]
    output_file_path = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\reg_result_1.json"
    output_file_path_2 = r"C:\Users\senbo\Desktop\taba_project\ai_series\result\reg_m_1.jpg"
    test_s_1.request(img_list, output_file_path, output_file_path_2)

start_reg()



