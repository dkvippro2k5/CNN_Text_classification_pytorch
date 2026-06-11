import pandas as pd
import os

# Tên file đầu vào và đầu ra
input_file = 'data.csv'
output_file = 'data_clean.csv'

def remove_duplicates():
    # 1. Kiểm tra file có tồn tại không
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file '{input_file}'")
        return

    print("Đang đọc file...")
    try:
        # Đọc file CSV (encoding utf-8 cho tiếng Việt)
        # Nếu file của bạn không có tiêu đề cột (header), hãy thêm: header=None
        df = pd.read_csv(input_file, encoding='utf-8')
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return

    # Lấy số lượng dòng ban đầu
    original_count = len(df)
    print(f"--> Tổng số dòng ban đầu: {original_count}")

    # 2. Xóa các dòng trùng lặp
    # keep='first': Giữ lại dòng đầu tiên tìm thấy, xóa các dòng trùng phía sau
    #df.drop_duplicates(inplace=True)
    
    # (Tùy chọn) Nếu bạn muốn lọc trùng lặp chỉ dựa trên cột nội dung (ví dụ cột 'text')
    df.drop_duplicates(keep='first', inplace=True)

    # Lấy số lượng dòng sau khi lọc
    new_count = len(df)
    removed_count = original_count - new_count

    print(f"--> Số dòng sau khi lọc: {new_count}")
    print(f"--> Đã xóa: {removed_count} dòng trùng lặp.")

    # 3. Lưu ra file mới
    try:
        # index=False để không lưu thêm cột số thứ tự 0,1,2...
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nThành công! File sạch đã được lưu tại: '{output_file}'")
        print("Bạn có thể đổi tên nó thành 'data.csv' để dùng cho train.")
    except Exception as e:
        print(f"Lỗi khi lưu file: {e}")

if __name__ == "__main__":
    remove_duplicates()