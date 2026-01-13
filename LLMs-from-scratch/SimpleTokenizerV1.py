
 class SimpleTokenizerV1:
     def __init__(self,vocab):
         self.str_to_id = vocab
         self.id_to_str = {v: k for k, v in vocab.items()}