import pickle

class ThuCnNewsSDataset:
    def __init__(self) -> None:
        self.pickle_path = "/home/leo/NLP/datasets/THUCNNEWS/THUNews_FULL.pickle"
        with open(self.pickle_path, "rb") as file:
            self.contents = pickle.load(file)
    
    def item_generator(self, start_file_index: int, start_item_index: int):
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