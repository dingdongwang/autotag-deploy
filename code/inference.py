import json
import os
import re
from functools import partial
import jieba
import pandas as pd
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from sklearn.metrics import classification_report
from transformers import BertForSequenceClassification
from transformers import BertModel
from transformers import BertTokenizer



batch_size=32


class Model(nn.Module):

    def __init__(self, pretrained_model, num_labels, pooling='cls'):
        super(Model, self).__init__()
        self.ptm = pretrained_model
        self.pooling = pooling
        self.classifier = nn.Linear(self.ptm.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.ptm(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            pooled_output = out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            pooled_output = out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            pooled_output = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            pooled_output = torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

        logits = self.classifier(pooled_output)
        return logits
    

label_data={'filling valve-water tap display': 0,
     'water level system-e18': 1,
     'heat exchanger-water leakage(broken)': 2,
     'heat exchanger-e09': 3,
     'inlet hose-blocked': 4,
     'heat pump-e05': 5,
     'inner door-deformation/damaged': 6,
     'control-power module-e09': 7,
     'fascia panel-damaged': 8,
     'heat exchanger-blocked': 9,
     'water level system-e14': 10,
     'control-power module-others': 11,
     'filling valve-broken': 12,
     'control-power module-program exception': 13,
     'drain hose-blocked': 14,
     'heat pump-e18': 15,
     'lower rack-part missing': 16,
     'heat exchanger-water tap display': 17,
     'drain pump-e24': 18,
     'door latch-door open during running': 19,
     'lower rack-paint dropping': 20,
     'fascia panel-loose': 21,
     'control-power module-e05': 22,
     'drain pump-e23': 23,
     'control-operating module-e20': 24,
     'drain pump-wireharness loose': 25,
     'door latch-noise': 26,
     'water level system-e17': 27,
     'filling valve-clip damage/loosen': 28,
     'fascia panel-paint dropping': 29,
     'control-power module-e22': 30,
     'control-operating module-display abnormal': 31,
     'fascia panel-e15': 32,
     'inlet hose-miscoding': 33,
     'door latch-not running': 34,
     'heat pump-leakage/poor connection/e15': 35,
     'inlet hose-e10': 36,
     'water level system-e24': 37,
     'lower rack-clamp damaged/fallen': 38,
     'control-operating module-e18': 39,
     'control-operating module-short circuit': 40,
     'control-power module-e00': 41,
     'heat exchanger-e19': 42,
     'heat pump-e04': 43,
     'upper rack-handle loose': 44,
     'heat exchanger-water leakage(connection)': 45,
     'heat exchanger-continue filling water': 46,
     'fascia panel-others': 47,
     'water level system-e11': 48,
     'control-operating module-e24': 49,
     'inner door-screw loosed': 50,
     'heat pump-e31': 51,
     'inner door-hinge fallen': 52,
     'lower rack-bearing danaged': 53,
     'drain pump-e25': 54,
     'control-power module-e06': 55,
     'filling valve-poor connection': 56,
     'control-power module-e28': 57,
     'water level system-e23': 58,
     'control-power module-e07': 59,
     'filling valve-others': 60,
     "door latch-can't open": 61,
     'control-power module-e14': 62,
     'control-power module-e11': 63,
     'dispenser-others': 64,
     'fascia panel-auto power on': 65,
     'inner door-not open': 66,
     'upper rack-others': 67,
     'control-operating module-e10': 68,
     'water level system-e15': 69,
     'lower rack-wheel damaged': 70,
     'door latch-others': 71,
     "door latch-can't close": 72,
     'drain pump-block,cleaning': 73,
     'heat pump-e02': 74,
     'control-power module-e18': 75,
     'dispenser-miscoding': 76,
     'heat pump-e22': 77,
     'inlet hose-strange smell': 78,
     'drain hose-leakage/e15': 79,
     'heat exchanger-e18': 80,
     'heat pump-e20': 81,
     'drain hose-damaged(hole)': 82,
     'control-power module-e04': 83,
     'fascia panel-button failure': 84,
     'control-power module-e03': 85,
     'fascia panel-deformation': 86,
     'control-power module-e34': 87,
     'additional acoustic parts-pvc fallen': 88,
     'inner door-leakage': 89,
     'water level system-damage': 90,
     "dispenser-detergent lid can't open": 91,
     'lower rack-corrosion/rust': 92,
     'door latch-e19': 93,
     'control-power module-e16': 94,
     'heat pump-blocked': 95,
     'control-operating module-e01': 96,
     'drain hose-e24': 97,
     'control-power module-e01': 98,
     'fascia panel-electric leakage': 99,
     'drain pump-e20': 100,
     'heat pump-e11': 101,
     'upper rack-wheel loosen': 102,
     'drain hose-miscoding': 103,
     'heat pump-e07': 104,
     'heat pump-e01': 105,
     'dispenser-poor connection': 106,
     'control-operating module-wrong coding': 107,
     'control-power module-e24': 108,
     'control-power module-e31': 109,
     'heat exchanger-leakage/poor connection/e15': 110,
     'inlet hose-buckled/bent/pinched': 111,
     'filling valve-not work': 112,
     'inner door-part missing': 113,
     'lower rack-wheel fallen': 114,
     'upper rack-wheel fallen': 115,
     'control-operating module-e15': 116,
     'inlet hose-wrong coding': 117,
     'drain hose-damaged': 118,
     'control-power module-short circuit': 119,
     'dispenser-e15': 120,
     'control-power module-e13': 121,
     'door latch-latch broken': 122,
     'drain hose-buckled/bent/pinched': 123,
     'water level system-others': 124,
     'upper rack-handle damaged': 125,
     'drain hose-loose(heat exchanger)': 126,
     'control-power module-water tap display': 127,
     'control-operating module-program exception': 128,
     'additional acoustic parts-pvc deformed': 129,
     'fascia panel-e23': 130,
     'control-power module-e25': 131,
     'drain pump-not drain': 132,
     'heat exchanger-others': 133,
     'filling valve-e25': 134,
     'heat pump-e24': 135,
     'additional acoustic parts-pvc damaged': 136,
     'additional acoustic parts-others': 137,
     'control-power module-e29': 138,
     'filling valve-e14': 139,
     'heat exchanger-e15': 140,
     'dispenser-e19': 141,
     'upper rack-rack rust': 142,
     'filling valve-e16': 143,
     'control-power module-wireharness loose': 144,
     'control-power module-water/matter entered': 145,
     'control-power module-e21': 146,
     'upper rack-part missing': 147,
     'control-power module-e26': 148,
     'dispenser-leakage': 149,
     'filling valve-e19': 150,
     'heat pump-e06': 151,
     'upper rack-wheel damaged': 152,
     'inner door-dirty': 153,
     'control-operating module-others': 154,
     'control-power module-e/f': 155,
     'heat exchanger-e14': 156,
     'heat pump-e09': 157,
     'lower rack-rack deformation/damaged': 158,
     'upper rack-wrong coding': 159,
     'inlet hose-connection  leakage(machine)': 160,
     'fascia panel-miscoding': 161,
     'filling valve-e24': 162,
     'dispenser-rinse aid leaking': 163,
     'control-operating module-incompleted display': 164,
     'control-power module-not work': 165,
     'control-power module-e20': 166,
     'heat exchanger-e13': 167,
     'water level system-water tap display': 168,
     'water level system-wireharness loose': 169,
     'door latch-e04': 170,
     'control-power module-e02': 171,
     "dispenser-detergent lid can't close": 172,
     'control-operating module-wireharness loose': 173,
     'filling valve-noise': 174,
     'heat pump-e23': 175,
     'upper rack-rack deformation/damaged': 176,
     'additional acoustic parts-pvc lossed': 177,
     'inlet hose-others': 178,
     'door latch-hinge/spring failure': 179,
     'inlet hose-damaged': 180,
     'dispenser-blocked': 181,
     'filling valve-miscoding': 182,
     'filling valve-e17': 183,
     'dispenser-wireharness loose': 184,
     'fascia panel-corrosion/rust': 185,
     'water level system-poor connection': 186,
     'heat exchanger-expansion nut loose': 187,
     'filling valve-wireharness loose': 188,
     'dispenser-rinse aid lid not open': 189,
     'lower rack-others': 190,
     'filling valve-block, adjust': 191,
     'heat pump-wireharness loose': 192,
     'control-operating module-water/matter entered': 193,
     'control-power module-e15': 194,
     'heat pump-not work': 195,
     'inlet hose-part missing': 196,
     'inlet hose-connection  leakage(tap)': 197,
     'heat pump-e12': 198,
     'heat exchanger-e17': 199,
     'heat pump-e19': 200,
     'heat pump-e30': 201,
     'dispenser-wrong coding': 202,
     'inlet hose-leakage': 203,
     'inner door-noise': 204,
     'inner door-corrosion/rust': 205,
     'upper rack-bearing damaged': 206,
     'control-operating module-button failure': 207,
     'drain pump-miscoding': 208,
     'inner door-others': 209,
     'water level system-e19': 210,
     'door latch-wrong coding': 211,
     'heat exchanger-wireharness loose': 212,
     'drain hose-no info': 213,
     'upper rack-wheel blocked': 214,
     'filling valve-continue filling water': 215,
     'water level system-e/f': 216,
     'lower rack-wheel blocked': 217,
     'dispenser-crack/deformation': 218,
     'water level system-wrong coding': 219,
     'heat pump-circuit breaker tripped': 220,
     'drain hose-others': 221,
     'control-power module-e19': 222,
     'drain pump-lid loose': 223,
     'drain pump-e15': 224,
     'control-operating module-poor connection': 225,
     'fascia panel-fog': 226,
     'inner door-miscoding': 227,
     'water level system-not work': 228,
     'upper rack-clamp damaged': 229,
     'filling valve-leakage/e15': 230,
     'inlet hose-e14': 231,
     'control-power module-e27': 232,
     'inner door-not close': 233,
     'control-power module-e10': 234,
     'drain pump-poor connection/leakage': 235,
     'heat exchanger-water leakage(flower meter)': 236,
     'drain pump-e26': 237,
     'drain hose-e14': 238,
     'drain pump-others': 239,
     'door latch-e15': 240,
     'control-power module-wrong coding': 241,
     'drain pump-e22': 242,
     'dispenser-rinse aid lid not close': 243,
     'heat pump-e21': 244,
     'heat pump-others': 245,
     'filling valve-e18': 246,
     'control-power module-e23': 247,
     'heat pump-noise': 248,
     'control-operating module-not work': 249,
     'drain hose-loose': 250,
     'drain pump-water tap display': 251,
     'drain pump-noise': 252}

class MyDataSet(Dataset):

    def __init__(self,text1,text2,text3,is_test=False):
        self.is_test = is_test
        self.data = pd.DataFrame({"Defect Found": text1,"Work Executed": text2,"QM Part Structure Text": text3},
                    index=[0])
        #self.data = pd.read_csv(file_path)

    def __getitem__(self, idx):
        if self.is_test:
            text_a, text_b, part = self.data.loc[idx]["Defect Found"], self.data.loc[idx][
                "Work Executed"], self.data.loc[idx]['QM Part Structure Text']
            part = part.strip().lower()
            if not (isinstance(text_a, str)):
                text_a = ""
            if not (isinstance(text_b, str)):
                text_b = ""
            text_a = re.sub(r"sj[0-9A-Za-z]*[-/]*\d*|SJ[0-9A-Za-z]*[-/]*\d*|//\d*-|\d*-\d*-", "", text_a)
            text_b = re.sub(r"</br>|br|\sbr\s|\s</br>\s", "。", text_b)
            return text_a, text_b, part, None
        else:
            return self.data.loc[idx]["text_a"], self.data.loc[idx]["text_b"],\
                   self.data.loc[idx]["part"], self.data.loc[idx]["label"]

    def __len__(self):
        return self.data.shape[0]
   




def collate_batch(batch, tokenizer, label_dict=None, is_test=False):
    text_a, text_b, labels = [], [], []
    for a, b, part, label in batch:
        text_a.append(part + " " + b)
        text_b.append(part + " " + a)
    encoded_inputs = tokenizer(text=text_a, text_pair=text_b, padding=True, return_tensors="pt")
    if not is_test and isinstance(label_dict, dict):
        labels = torch.tensor([label_dict[item[3]] for item in batch])
        return encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], encoded_inputs["attention_mask"], labels
    else:
        return encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], encoded_inputs["attention_mask"]
    
    
    
def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ptm = BertModel.from_pretrained("nghuyong/ernie-gram-zh", num_labels=253)
    model = Model(ptm, num_labels=253, pooling='last-avg')
    with open(os.path.join(model_dir, 'epoch_9.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f,map_location=device))

    return model




def input_fn(request_body,request_content_type):
    #assert request_content_type == "application/json"
    #request_body=json.dumps(request_body)
    if request_content_type == "application/json":
        data = json.loads(request_body)
    else:
        data=request_body

    return data



def predict_fn(input_object, model):
    label_dict=label_data
    inv_label = {value: key for key, value in label_dict.items()}  
    
    text1=input_object['defectFound']
    text2 = input_object['workExecuted']
    text3 = input_object['qmPartStructureText']
    
    pred_ds = MyDataSet(text1,text2,text3,is_test=True)
    tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-gram-zh")
    trans_fn = partial(collate_batch, tokenizer=tokenizer, label_dict=label_dict, is_test=True)
    data_loader = DataLoader(pred_ds, shuffle=False, batch_size=batch_size, collate_fn=trans_fn)

    tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-gram-zh")

    with torch.no_grad():
            for idx, (input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
                input_ids=input_ids
                token_type_ids=token_type_ids
                attention_mask=attention_mask    
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    results = []
    confs = []
    with torch.no_grad():
        #for idx, (input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        ttention_mask = attention_mask.to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        max_probs = max_probs.cpu().numpy().tolist()
        preds = torch.argmax(logits, dim=-1)
        preds = preds.cpu().numpy().tolist()
        results.extend([inv_label[item] for item in preds])
        confs.extend(max_probs)

    prediction=[results,confs]
    return results,confs #prediction




def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    
    if response_content_type == "application/json":
        response = json.dumps(prediction) #str(prediction)
    else:
        response = prediction #str(prediction)

    return response