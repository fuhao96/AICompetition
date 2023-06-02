
from config import parsers
# transformer库是一个把各种预训练模型集成在一起的库，导入之后，你就可以选择性的使用自己想用的模型，这里使用的BERT模型。
# 所以导入了bert模型，和bert的分词器，这里是对bert的使用，而不是bert自身的源码。
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


def read_data(file):
    # 读取文件
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    # 得到所有文本、所有标签、句子的最大长度
    texts, labels, max_length = [], [], []
    for data in all_data:
        if data:
            text, label = data.split("\t")
            max_length.append(len(text))
            texts.append(text)
            labels.append(label)

    return texts, labels


class MyDataset(Dataset):
    def __init__(self, texts, labels, with_labels=True):
        self.all_text = texts
        self.all_label = labels
        self.max_len = parsers().max_len
        self.with_labels = with_labels
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
        
        # 定义文本标签到整数标签的映射关系
        self.label_map = {'文学': 0, '童书': 1, '工业技术': 2, '大中专教材教辅': 3, '中小学教辅': 4, '艺术': 5, '管理': 6, '社会科学': 7, '建筑': 8, '小说': 9, '计算机与互联网': 10, '历史': 11, '法律': 12, '医学': 13, '科学与自然': 14, '外语学习': 15, '政治/军事': 16, '考试': 17, '哲学/宗教': 18, '励志与成功': 19, '经济': 20, ' 文化': 21, '青春文学': 22, '传记': 23, '农业/林业': 24, '动漫': 25, '烹饪/美食': 26, '旅游/地图': 27, '育儿/家教': 28, '科普读物': 29, '国学/古籍': 30, '孕产/胎教': 31, '健身与保健': 32, '金融与投资': 33, '婚恋与两性': 34}  # 根据实际情况进行修改

    def __getitem__(self, index):
        text = self.all_text[index]
        
        # 使用label_map将文本标签转换为整数标签
        label_text = self.all_label[index]
        label = self.label_map[label_text]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(text,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.max_len,
                                      return_tensors='pt')

        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

        if self.with_labels:
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

    def __len__(self):
        return len(self.all_text)


if __name__ == "__main__":
    train_text, train_label = read_data("./data/train.txt")
    print(train_text[0], train_label[0])
    trainDataset = MyDataset(train_text, labels=train_label, with_labels=True)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)
    for i, batch in enumerate(trainDataloader):
        print(batch[0], batch[1], batch[2], batch[3])
