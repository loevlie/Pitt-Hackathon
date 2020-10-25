from tkinter import *
from tkhtmlview import HTMLLabel
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


tr=pd.read_csv('./healthcare/train_data.csv')
di=pd.read_csv('./healthcare/train_data_dictionary.csv')
df_test=pd.read_csv('./healthcare/test_data.csv')
d={}
for x in di['Column'][:-1]:
    ll=tr[x].unique()
    if len(ll)<50:
        d.update({x:ll})  
    else:
        d.update({x:'-'})

s=[]

class Model(nn.Module):

    def __init__(self,in_features=74,h1=2048,h2=2048,h3=1024*2,h4=1024,h5=900,h6=900,h7=800,
                 h8=800,h9 = 800,h10=800,h11=800,h12=800,h13=800,h14=800,h15=800,out_features=11):
        
        # How many layers?
        # Input layer (# of features) --> hidden layer 1 (number of neurons N) --> h2 (N) --> output (346 of classes)
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.bn1 = nn.BatchNorm1d(num_features=h1,momentum=0.01)
        self.fc2 = nn.Linear(h1,h2)
        self.d2 =  nn.Dropout(0.25)
        self.bn2 = nn.BatchNorm1d(num_features=h2,momentum=0.01)
        self.fc3 = nn.Linear(h2,h3)
        self.bn3 = nn.BatchNorm1d(num_features=h3,momentum=0.01)
        self.d3 = nn.Dropout(0.4)
        self.fc4 = nn.Linear(h3,h4)
        self.bn4 = nn.BatchNorm1d(num_features=h4,momentum=0.01)
        self.d4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(h4,h5)
        self.bn5 = nn.BatchNorm1d(num_features=h5,momentum=0.01)
        self.d5 = nn.Dropout(0.25)
        self.fc6 = nn.Linear(h5,h6)
        self.bn6 = nn.BatchNorm1d(num_features=h6,momentum=0.01)
        self.d6 = nn.Dropout(0.35)
        
        self.fc7 = nn.Linear(h6,h7)
        self.bn7 = nn.BatchNorm1d(num_features=h7,momentum=0.01)
        self.d7 = nn.Dropout(0.4)
        
        self.fc8 = nn.Linear(h7,h8)
        self.bn8 = nn.BatchNorm1d(num_features=h8,momentum=0.01)
        self.d8 = nn.Dropout(0.35)
        
        self.fc9 = nn.Linear(h8,h9)
        self.bn9 = nn.BatchNorm1d(num_features=h9,momentum=0.01)
        self.d9 = nn.Dropout(0.2)
        
        self.fc10 = nn.Linear(h9,h10)
        self.bn10 = nn.BatchNorm1d(num_features=h10,momentum=0.01)
        self.d10 = nn.Dropout(0.25)
        
        self.fc11 = nn.Linear(h10,h11)
        self.bn11 = nn.BatchNorm1d(num_features=h11,momentum=0.01)
        self.d11 = nn.Dropout(0.2)

        self.fc12 = nn.Linear(h11,h12)
        self.bn12 = nn.BatchNorm1d(num_features=h12,momentum=0.01)
        self.d12 = nn.Dropout(0.2)

        self.fc13 = nn.Linear(h12,h13)
        self.bn13 = nn.BatchNorm1d(num_features=h13,momentum=0.01)
        self.d13 = nn.Dropout(0.2)

        self.fc14 = nn.Linear(h13,h14)
        self.bn14 = nn.BatchNorm1d(num_features=h14,momentum=0.01)
        self.d14 = nn.Dropout(0.2)

        self.fc15 = nn.Linear(h14,h15)
        self.bn15 = nn.BatchNorm1d(num_features=h15,momentum=0.01)
        self.d15 = nn.Dropout(0.2)

        self.out = nn.Linear(h15,out_features)
  
    def forward(self,x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.d2(self.fc2(x))))
        x = F.relu(self.bn3(self.d3(self.fc3(x))))
        x = F.relu(self.bn4(self.d4(self.fc4(x))))
        x = F.relu(self.bn5(self.d5(self.fc5(x))))
        x = F.relu(self.bn6(self.d6(self.fc6(x))))
        x = F.relu(self.bn7(self.d7(self.fc7(x))))
        x = F.relu(self.bn8(self.d8(self.fc8(x))))
        x = F.relu(self.bn9(self.d9(self.fc9(x))))
        x = F.relu(self.bn10(self.d10(self.fc10(x))))
        x = F.relu(self.bn11(self.d11(self.fc11(x))))
        x = F.relu(self.bn12(self.d12(self.fc12(x))))
        x = F.relu(self.bn13(self.d13(self.fc13(x))))
        x = F.relu(self.bn14(self.d14(self.fc14(x))))
        x = F.relu(self.bn15(self.d15(self.fc15(x))))

        x = self.out(x)
        return x
    
model2 = Model()
device = torch.device("cpu")
model2.load_state_dict(torch.load('MyModel_Healthcare_final.pt',map_location=device));
model2.eval()
model2.to(device)
def get_pred(model,df):
    model.eval()
    stay = []
    def PreProcess(df):
        df['City_Code_Patient'] = df['City_Code_Patient'].astype(str).astype(float)
        df['Bed Grade'] = df['Bed Grade'].astype(str).astype(float)
        df['City_Code_Patient'].fillna(-99,inplace=True)
        df['Bed Grade'].fillna(5,inplace=True)
        wards = pd.get_dummies(df['Ward_Type'],drop_first=True)
        Age = pd.get_dummies(df['Age'],drop_first=True)
        df['Hospital_code'] = df['Hospital_code'].astype(str).astype(float)
        hospital_code =  pd.get_dummies(df['Hospital_code'],drop_first=True)
        hospital_type_code =  pd.get_dummies(df['Hospital_type_code'],drop_first=True)
        hospital_region_code =  pd.get_dummies(df['Hospital_region_code'],drop_first=True)
        department = pd.get_dummies(df['Department'],drop_first=True)
        ward_facility_code = pd.get_dummies(df['Ward_Facility_Code'],drop_first=True)
        bed_grade = pd.get_dummies(df['Bed Grade'],drop_first=True)
        admission_type = pd.get_dummies(df['Type of Admission'],drop_first=True)
        severity = pd.get_dummies(df['Severity of Illness'],drop_first=True)
        df = pd.concat([df,Age,wards,hospital_code,hospital_type_code,hospital_region_code,department,ward_facility_code,bed_grade,admission_type,severity],axis = 1)
        
        df.drop(['case_id','Age','Ward_Type','Hospital_code','Hospital_type_code','Hospital_region_code','Department','City_Code_Hospital','Ward_Facility_Code','Bed Grade','Type of Admission','Severity of Illness','patientid'],axis=1,inplace=True)
        
        df['Visitors with Patient']=2
        df['Admission_Deposit']=5000.0
        df['Available Extra Rooms in Hospital']= df['Available Extra Rooms in Hospital'].astype(str).astype(float)
        print(df.head())
        print(df.to_numpy()[-1])
        x = torch.Tensor(df.to_numpy()[-1])
        return x

    df = pd.concat([df_test,df])
    Input = PreProcess(df)
    Input = Input.resize(1,Input.shape[0])
    _,output = torch.max(F.softmax(model(Input),dim=1),1)
    convert = {0: '0-10',1: '11-20',2: '21-30',3: '31-40',4: '41-50',5: '51-60',6: '61-70',7: '71-80',8: '81-90',9: '91-100',10: 'More than 100 Days'}
    out = convert[int(output[0])]
    return out

def press():
    df={dk[ix]:s[ix] for ix in range(len(s))}
    point=pd.DataFrame(df,index=[0])
    """
    Insert Neural Network file
    Point has the same format as the input dataset
    Predict for this point
    Output it the way you want to
    """
    pred = get_pred(model2,point)
    #print(f'The patient will stay roughly {pred} days')
    label=Label(root,text=f'The patient should ideally stay for \n{pred} days',borderwidth=2, relief="groove")
    label.grid(row=10,column=4)
def create_dropdown(name,lis,root,mainframe):
    tkvar=StringVar(root)
#    l=sorted(lis)
    popupmenu=OptionMenu(mainframe,tkvar,*lis)
    tkvar.set('')
    tkvar.trace_add('write', lambda *args: s.append(tkvar.get()))
    w=Label(mainframe,text=name)
    return popupmenu,w,tkvar

#dk=sorted(d.keys())
dk=list(d.keys())
root = Tk()
root.title("Stay duration estimator")
mainframe = Frame(root)
mainframe.grid(column=0,row=0, sticky=(N,W,E,S) )
mainframe.columnconfigure(0, weight = 1)
mainframe.rowconfigure(0, weight = 1)

pp=[]
tkk=[]

for i,xx in enumerate(dk):
    w,x,tkv=create_dropdown(xx,d[xx],root,mainframe)
    tkk.append(tkv)
    pp.append([x,w])
count=0
for j,ls in enumerate(pp):
    if count<16:
        count=count+1
        x,w=ls
        x.grid(row=count,column=1)
        count=count+1
        w.grid(row=count,column=1)
    else:
        count=count+1
        x,w=ls
        x.grid(row=count-16,column=2)
        count=count+1
        w.grid(row=count-16,column=2)

B = Button(root, text ="Predict",command=press,font=("Times New Roman", 20)).grid(row=count+1,column=0)

root.mainloop()
