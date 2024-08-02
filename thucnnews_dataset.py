import pickle
from typing import List

class ThuCnNewsSDataset:
    def __init__(self) -> None:
        self.pickle_path = "~/THUNews_FULL.pickle"
        with open(self.pickle_path, "rb") as file:
            self.contents: List[str] = pickle.load(file)
    
    def item_generator(self, start_file_index: int, start_item_index: int):
        # 这个start_file_index属于我的垃圾代码，不用在意。
        # 在早期阶段直接用的悟道等数据集的压缩文件，它是为了记录读到哪个文件了
        index = start_item_index
        
        while True:
            yield {"fileno": start_file_index, "entryno": index, "content": self.contents[index]}
            if index < len(self.contents) - 1:
                index += 1
            else:
                index = 0
                start_file_index += 1

if __name__=="__main__":
    dataset = ThuCnNewsSDataset()
    print(len(dataset.contents))