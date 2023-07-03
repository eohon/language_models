import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# άνοιγμα αρχείου dataset
dataset = pd.read_csv('data.csv')

# επιλογή στηλών από το dataset 
keep = ['desc']
dataset = dataset[keep]

# διαγραφή κενών σειρών
dataset = dataset.dropna()

# εκτέλεση tokenization και vectorization 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenized_t = []
input_datum = []
masks = []

for index, row in dataset.iterrows():
    text = str(row['desc'])  # μετατρέπω σε string για να είμαι σίγουρος

    # Tokenize the text
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_mask=True,
        return_tensors='pt'
    )

    # παίρνω την είσοδο από την μάσκα
    input_id = encoded_text['input_datum'].flatten()
    mask = encoded_text['mask'].flatten()

    # εισάγω τα tokens και την είσοδο (input_datum) στις αντίστοιχες λίστες
    tokenized_t.append(text)
    input_datum.append(input_id)
    masks.append(mask)

# μετατρέπω τις λίστες κατάλληλα 
input_datum = torch.stack(input_datum)
masks = torch.stack(masks)

# φορτώνω το BERT
model = BertModel.from_pretrained('bert-base-uncased')

with torch.no_grad():
    outputs = model(input_datum, mask=masks)
    embeddings = outputs.last_hidden_state

