import re
import jieba
from tqdm import tqdm
from typing import List, Optional


class TextPreprocessor:

    def __init__(self, custom_dict_path: Optional[str] = None, stopwords_path: str = 'stopwords.txt'):
       
        self.custom_dict_path = custom_dict_path
        self.stopwords_path = stopwords_path
        self.stopwords = set()
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
        self._load_stopwords()

    def _load_stopwords(self):
        try:
            with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f if line.strip()])
            print(f"[OK] 加载停用词: {len(self.stopwords)} 个")
        except FileNotFoundError:
            print(f"[警告] 停用词文件不存在: {self.stopwords_path}")
            self.stopwords = set()

    def clean_text(self, text: str) -> str:
       
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def tokenize(self, text: str, use_stopwords: bool = True) -> List[str]:
       
        tokens = list(jieba.cut(text))
        if use_stopwords:
            tokens = [token for token in tokens
                      if token not in self.stopwords
                      and len(token.strip()) > 0
                      and not token.strip().isspace()]
        else:
            tokens = [token for token in tokens
                      if len(token.strip()) > 0
                      and not token.strip().isspace()]

        return tokens

    def preprocess(self, text: str, use_stopwords: bool = True) -> List[str]:
      
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text, use_stopwords=use_stopwords)

        return tokens

    def preprocess_batch(self, texts: List[str], use_stopwords: bool = True, show_progress: bool = True) -> List[List[str]]:
       
        tokens_list = []

        if show_progress:
            texts = tqdm(texts, desc="  预处理文本")

        for text in texts:
            tokens = self.preprocess(text, use_stopwords=use_stopwords)
            tokens_list.append(tokens)

        return tokens_list
