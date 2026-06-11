# Tạo file check_fix.py cùng thư mục main.py
from main import text_field

print("------------------------------------------------")
if text_field.batch_first == True:
    print("ĐÃ SỬA ĐÚNG! (batch_first=True)")
    print("Bạn có thể chạy 'python main.py' ngay.")
else:
    print("VẪN SAI! (batch_first đang là False)")
    print("Hãy quay lại main.py và thêm 'batch_first=True' vào dòng data.Field")
print("------------------------------------------------")