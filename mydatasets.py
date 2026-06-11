import re
from torchtext import data
try:
    from underthesea import word_tokenize
except ImportError:
    print("Chua cai underthesea")

# Hàm Tokenizer
def vietnamese_tokenizer(text):
    text = re.sub(r"[^\w\s(),!?]", " ", text)
    return word_tokenize(text, format="text").split()

class MyCSVDataset(data.TabularDataset):
    
    @classmethod
    def splits(cls, text_field, label_field, path=r'D:\cnn-text-classification-pytorch\data.csv', train='data.csv', **kwargs):
        
        fields = [
            ('id', None),           # Cột 0: Bỏ qua
            ('label', label_field), # Cột 1: Nhãn
            ('text', text_field)    # Cột 2: Nội dung
        ]
        
        # Tạo dataset
        dataset = cls(
            path=f"{path}/{train}", 
            format='csv', 
            fields=fields, 
            # Nếu dòng đầu tiên của file là dữ liệu luôn (0,sales...) thì skip_header=False
            # Nếu dòng đầu tiên là tiêu đề (id,category,content) thì skip_header=True
            skip_header=False 
        )
        
        import random
        random.seed(1234)
        return dataset.split(split_ratio=[0.8, 0.1, 0.1], random_state=random.getstate())