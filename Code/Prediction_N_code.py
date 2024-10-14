# -*- coding: utf-8 -*-
"""
Created on Thu May 23 20:40:29 2024

@author: jiaxin Huang
"""
#Code written by: Shuya Guo

##General overview
##Pred_D is an interactive desktop application for predicting diffusion coefficient of gas molecules in MOF or other porous crystal materials.The core of its computation is our trained LGBM model. The following is the source code for the interface design.

## Import the required libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor as lgb
import tkinter as tk 
import ttkbootstrap as ttk
from tkinter import filedialog
from PIL import Image,ImageTk
import webbrowser
import joblib
import sys
import os

## Load the trained LGBM model
model = joblib.load("../model/lgbm.pt") ## The detailed code for model training is in “LGBM_code.py”

## The layout design of the main interface
root= tk.Tk()
root.title("Predict adsorption  of material on LGBM")
root.resizable(True,True) ## The window size can be changed
canvas1 = tk.Canvas(root, width = 820, height =730) ## Main window size
#canvas1.config(bg='Light blue')
canvas1.pack()

## The plate layout within the main interface
re1=canvas1.create_rectangle(50,20,770,450)
re2=canvas1.create_rectangle(50,470,770,670)
re3=canvas1.create_rectangle(70,100,380,320,outline='darkgray')
re4=canvas1.create_rectangle(400,100,750,430,outline='darkgray')
re4=canvas1.create_rectangle(70,340,380,430,outline='darkgray')
re4=canvas1.create_rectangle(70,560,750,650,outline='darkgray')
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(250, 340, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',10),text='Predicted results')
canvas1.create_window(620, 560, window=label_B)
label_B = tk.Label(root,font=('microsoft yahei',9),text='Author：Jiaxin Huang,Zhiwei Qiao,Guangzhou University')
canvas1.create_window(580, 715, window=label_B)

## Message box (Related literature on molecular physical properties)
def cmx1():
    window = tk.Tk()     
    window.title('Warm prompt')     
    window.geometry('350x250')
    link = tk.Label(window, text='The physical properties of gases \nare known from the literature:\nhttps://doi.org/10.1039/B802426J'
                    , font=('microsoft yahei',10),anchor="center")
    link.place(x=30, y=50) 
    def open_url(event):
        webbrowser.open("https://doi.org/10.1039/B802426J", new=0)         
    link.bind("<Button-1>", open_url)    
btn1=tk.Button(root, text='Tooltip',font=('microsoft yahei',10), command=cmx1)
btn1.configure(bg='orange')
canvas1.create_window(450, 140, window=btn1)

##Message box (Instructions for Prediction of a single material diffusiivity)
def resize(w, h, w_box, h_box, pil_image): 
  f1 = 1*w_box/w 
  f2 =1*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height),Image.ANTIALIAS)
       
w_box = 600  
h_box = 450    

global tk_image 
photo1 = Image.open("../Img/full_name.png")
w, h = photo1.size       
photo1_resized = resize(w, h, w_box, h_box, photo1)    
tk_image1 = ImageTk.PhotoImage(photo1_resized)

def cmx2():
    top2=tk.Toplevel() 
    top2.title('Instructions for use') 
    top2.geometry('620x500') 
    lab_1 = ttk.Label(top2,image=tk_image1) 
    lab_1.place(x=25, y=10) 
    top2.mainloop()  
  
btn2=tk.Button(root, text='README',font=('microsoft yahei',10), command=cmx2)
btn2.configure(bg='orange')
canvas1.create_window(120, 60, window=btn2)

## Message box (Instructions for batch Prediction of material diffusiivity)
def resize(w, h, w_box, h_box, pil_image):
  f1 = 1*w_box/w 
  f2 =1*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height), Image.ANTIALIAS)      
w_box = 600  
h_box = 500    

global tk_image 
photo2 = Image.open("../Img/sample_file.png")  
w, h = photo2.size     
photo2_resized = resize(w, h, w_box, h_box, photo2)    
tk_image2 = ImageTk.PhotoImage(photo2_resized)

def cmx3():
    top1=tk.Toplevel()
    top1.title('Instructions for use')     
    top1.geometry('680x580')
    lab2 = tk.Label(top1, text='You need to create the data you want to compute\nin the format below (For example：the prediction \nof Nn-C5H12):'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab2.place(x=20, y=20) 
    lab3 = tk.Label(top1, text='After creating the file, you can click the import file \nbutton on the screen.The predicted result  will be \nsaved in "Result/Batch_Predicted_N.xlsx".'
                    , font=('microsoft yahei',15),anchor="nw",justify='left')
    lab3.place(x=30, y=450)
    lab3 = ttk.Label(top1,text="photo:",image=tk_image2)
    lab3.place(x=30, y=120) 
    top1.mainloop()     
btn3=tk.Button(root, text='README',font=('microsoft yahei',10),command=cmx3)
btn3.configure(bg='orange')
canvas1.create_window(120, 520, window=btn3)

## Sets the label and entry for entering the nine descriptor 
label_Z = tk.Label(root,font=('microsoft yahei',13),text='Predict adsorption of material')
canvas1.create_window(415, 20, window=label_Z)

label_L = tk.Label(root,font=('microsoft yahei',11),text='Physical property of material')
canvas1.create_window(225, 100, window=label_L)

label1 = tk.Label(root,font=('microsoft yahei',10),text='HVF：') ## create 1st label box 
canvas1.create_window(160, 140, window=label1)
entry1 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 1st entry box 
canvas1.create_window(300, 140, window=entry1)

label2 = tk.Label(root,font=('microsoft yahei',10), text='PLD (Å): ') ## create 2st label box 
canvas1.create_window(160, 180, window=label2)
entry2 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 2nd entry box
canvas1.create_window(300, 180, window=entry2)

label3 = tk.Label(root,font=('microsoft yahei',10), text='LCD (Å): ') ## create 3st label box 
canvas1.create_window(160, 220, window=label3)
entry3 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 3nd entry box
canvas1.create_window(300, 220, window=entry3)

label4 = tk.Label(root,font=('microsoft yahei',10,"italic"), text='ρ') ## create 4st label box 
canvas1.create_window(110, 260, window=label4)
l0 = tk.Label(root,font=('microsoft yahei',10), text='(kg/cm^3): ') 
canvas1.create_window(170, 260, window=l0)
entry4 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 4nd entry box
canvas1.create_window(300, 260, window=entry4)

label5 = tk.Label(root, font=('microsoft yahei',10),text='VSA (m^2/cm^3): ') ## create 5st label box 
canvas1.create_window(160, 300, window=label5) 
entry5 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 5nd entry box
canvas1.create_window(300, 300, window=entry5)

label_R = tk.Label(root,font=('microsoft yahei',11),text='Physical property of gas molecules') 
canvas1.create_window(575, 100, window=label_R)

label6 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dia') ## create 1st 6abel box 
canvas1.create_window(490,180, window=label6)
l1 = tk.Label(root, font=('microsoft yahei',10),text='(Å):')
canvas1.create_window(525,180, window=l1) 
entry6 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 6nd entry box
canvas1.create_window(660,180, window=entry6)

label7 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Pol') ## create 7st label box 
canvas1.create_window(430,220, window=label7)
l2 = tk.Label(root, font=('microsoft yahei',10,),text='(×10^25/cm^3): ')
canvas1.create_window(520,220, window=l2)
entry7 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 7nd entry box
canvas1.create_window(660,220, window=entry7)

label8 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Dip') ## create 8st label box 
canvas1.create_window(485,260, window=label8) 
l3 = tk.Label(root, font=('microsoft yahei',10),text='(D): ')
canvas1.create_window(525,260, window=l3)  
entry8 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 8nd entry box
canvas1.create_window(660, 260, window=entry8)

label9 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Tb') ## create 9st label box 
canvas1.create_window(485, 300, window=label9)
l4 = tk.Label(root, font=('microsoft yahei',10),text='(K): ')
canvas1.create_window(520, 300, window=l4) 
entry9 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 9nd entry box
canvas1.create_window(660,300, window=entry9)

label10 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Tc') ## create 9st label box 
canvas1.create_window(485, 340, window=label10)
l5 = tk.Label(root, font=('microsoft yahei',10),text='(K): ')
canvas1.create_window(520, 340, window=l5) 
entry10 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 9nd entry box
canvas1.create_window(660,340, window=entry10)

label11 = tk.Label(root, font=('microsoft yahei',10,"italic"),text='Pc') ## create 9st label box 
canvas1.create_window(480, 380, window=label11)
l6 = tk.Label(root, font=('microsoft yahei',10),text='(bar): ')
canvas1.create_window(520, 380, window=l6) 
entry11 = tk.Entry (root,font=('microsoft yahei',10),width=12,justify='center') ## create 9nd entry box
canvas1.create_window(660,380, window=entry11)

## The linkage between the physical properties of molecules and molecules is realized
## Sets four properties: 1-dia,2-pol,3-Dip,4-Qua
input1 = 0
input2 = 0
input3 = 0
input4 = 0
input5 = 0
input6 = 0
def run2():
    dic1 = {0: 'Dia', 1: 'Pol', 2: 'Dip',3: 'Tb',4:'Tc',5:'Pc'}
    c2 = dic1[cm1.current()]
    if (c2 == 'Dia'):                      
        entry6 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input1,justify='center')
        canvas1.create_window(660,180, window=entry6)              
    elif (c2 =='Pol'):
        entry7 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input2,justify='center') 
        canvas1.create_window(660,220, window=entry7)
    elif (c2 =='Dip'):
        entry8 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input3,justify='center') 
        canvas1.create_window(660, 260, window=entry8)                   
    elif (c2 =='Tb'):
        entry9 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(660,300, window=entry9)
    elif (c2 =='Tc'):
        entry10 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(660,340, window=entry10)
    elif (c2 =='Pc'):
        entry11 = tk.Entry (root,font=('microsoft yahei',10),width=12,text=input4,justify='center') 
        canvas1.create_window(660,380, window=entry11)
        
def calc(event):
    global input1
    global input2
    global input3
    global input4  
    global input5 
    global input6    
    dic = {0: 'HCHO', 1: 'C2H5OH', 2: 'C3H8',3: 'n-C4H10',4: 'n-C5H12',5: 'n-C6H14',6: 'C8H10',7: 'C4H9ClS',8: 'C3H9O3P',9:'C4H8Cl2S',10:'C7H16FO2P'}
    c = dic[cm1.current()] 
    if (c == 'HCHO'): 
        input1 = 2.7 
        input2 = 25.9
        input3 = 2.332 
        input4 = 252.15
        input5 = 408
        input6 = 65.9
        entry6.delete(0,"end")
        entry6.insert(0,"2.7")
        entry7.delete(0,"end")
        entry7.insert(0,"25.9")
        entry8.delete(0,"end")
        entry8.insert(0,"2.332")
        entry9.delete(0,"end")
        entry9.insert(0,"252.15")
        entry10.delete(0,"end")
        entry10.insert(0,"408")
        entry11.delete(0,"end")
        entry11.insert(0,"65.9") 
    elif (c == 'C2H5OH'):
        input1 = 4.53
        input2 = 52.6
        input3 = 1.69
        input4 = 351.8
        input5 = 513.92
        input6 = 61.48 
        entry6.delete(0,"end")
        entry6.insert(0,"4.53")
        entry7.delete(0,"end")
        entry7.insert(0,"52.6")
        entry8.delete(0,"end")
        entry8.insert(0,"1.69")
        entry9.delete(0,"end")
        entry9.insert(0,"351.8")
        entry10.delete(0,"end")
        entry10.insert(0,"513.92")
        entry11.delete(0,"end")
        entry11.insert(0,"61.48")        
    elif (c == 'C3H8'):
        input1 = 4.709
        input2 = 63.3
        input3 = 0.084
        input4 = 231.02
        input5 = 369.83
        input6 = 42.48
        entry6.delete(0,"end")
        entry6.insert(0,"4.709")
        entry7.delete(0,"end")
        entry7.insert(0,"63.3")
        entry8.delete(0,"end")
        entry8.insert(0," 0.084")
        entry9.delete(0,"end")
        entry9.insert(0,"231.02")
        entry10.delete(0,"end")
        entry10.insert(0,"369.83")
        entry11.delete(0,"end")
        entry11.insert(0,"42.48")  
    elif (c == 'n-C4H10'):
        input1 = 4.687
        input2 = 82
        input3 = 0.05
        input4 = 272.66
        input5 = 425.12 
        input6 = 37.96
        entry6.delete(0,"end")
        entry6.insert(0,"4.687")
        entry7.delete(0,"end")
        entry7.insert(0,"82")
        entry8.delete(0,"end")
        entry8.insert(0,"0.05")
        entry9.delete(0,"end")
        entry9.insert(0,"272.66")
        entry10.delete(0,"end")
        entry10.insert(0,"425.12")
        entry11.delete(0,"end")
        entry11.insert(0,"37.96") 
    elif (c == 'n-C5H12'):
        input1 = 4.5
        input2 = 99.9
        input3 = 0
        input4 = 309.22
        input5 = 469.7 
        input6 = 33.7
        entry6.delete(0,"end")
        entry6.insert(0,"4.5")
        entry7.delete(0,"end")
        entry7.insert(0,"99.9")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"309.22")  
        entry10.delete(0,"end")
        entry10.insert(0,"469.7 ")
        entry11.delete(0,"end")
        entry11.insert(0,"33.7") 
    elif (c == 'n-C6H14'):
        input1 = 4.3
        input2 = 119
        input3 = 0
        input4 = 341.88 
        input5 = 507.6 
        input6 = 30.25
        entry6.delete(0,"end")
        entry6.insert(0,"4.3")
        entry7.delete(0,"end")
        entry7.insert(0,"119")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"341.88") 
        entry10.delete(0,"end")
        entry10.insert(0,"507.6")
        entry11.delete(0,"end")
        entry11.insert(0,"30.25")
    elif (c == 'C8H10'):
        input1 = 6.467 
        input2 = 143.33
        input3 = 0.37
        input4 = 413.82
        input5 = 619.17
        input6 = 35.95 
        entry6.delete(0,"end")
        entry6.insert(0,"6.467")
        entry7.delete(0,"end")
        entry7.insert(0,"143.33")
        entry8.delete(0,"end")
        entry8.insert(0,"0.37")
        entry9.delete(0,"end")
        entry9.insert(0,"413.82") 
        entry10.delete(0,"end")
        entry10.insert(0,"619.17")
        entry11.delete(0,"end")
        entry11.insert(0,"35.95")
    elif (c == 'C4H9ClS'):
        input1 = 5.8
        input2 = 132.5
        input3 = 2.25
        input4 = 429.22
        input5 = 540.13
        input6 = 27.36
        entry6.delete(0,"end")
        entry6.insert(0,"5.8")
        entry7.delete(0,"end")
        entry7.insert(0,"132.5")
        entry8.delete(0,"end")
        entry8.insert(0,"2.25")
        entry9.delete(0,"end")
        entry9.insert(0,"429.22") 
        entry10.delete(0,"end")
        entry10.insert(0,"540.13")
        entry11.delete(0,"end")
        entry11.insert(0,"27.36")
    elif (c == 'C3H9O3P'):
        input1 = 5.7
        input2 = 104.3
        input3 = 2.27
        input4 = 458.4
        input5 = 700.6
        input6 = 49.7
        entry6.delete(0,"end")
        entry6.insert(0,"5.7")
        entry7.delete(0,"end")
        entry7.insert(0,"104.3")
        entry8.delete(0,"end")
        entry8.insert(0,"2.27")
        entry9.delete(0,"end")
        entry9.insert(0,"458.4")   
        entry10.delete(0,"end")
        entry10.insert(0,"700.6")
        entry11.delete(0,"end")
        entry11.insert(0,"49.7")
    elif (c == 'C4H8Cl2S'):
        input1 = 5.9
        input2 = 158.3
        input3 = 1.182
        input4 = 489.15
        input5 = 540.13
        input6 = 27.36
        entry6.delete(0,"end")
        entry6.insert(0,"5.9")
        entry7.delete(0,"end")
        entry7.insert(0,"158.3")
        entry8.delete(0,"end")
        entry8.insert(0,"1.182")
        entry9.delete(0,"end")
        entry9.insert(0,"489.15")  
        entry10.delete(0,"end")
        entry10.insert(0,"540.13")
        entry11.delete(0,"end")
        entry11.insert(0,"27.36")
    elif (c == 'C7H16FO2P'):
        input1 = 7.2
        input2 = 171.2
        input3 = 0
        input4 = 467.6
        input5 = 674.9
        input6 = 29.2
        entry6.delete(0,"end")
        entry6.insert(0,"7.2")
        entry7.delete(0,"end")
        entry7.insert(0,"171.2")
        entry8.delete(0,"end")
        entry8.insert(0,"0")
        entry9.delete(0,"end")
        entry9.insert(0,"467.6")   
        entry10.delete(0,"end")
        entry10.insert(0,"674.9")
        entry11.delete(0,"end")
        entry11.insert(0,"29.2")
## Create a Drop-down box
var1 = tk.StringVar() ## Create a variable 
cm1 = ttk.Combobox(root, textvariable=var1,font=('microsoft yahei',10)) ## Create a drop-down menu
cm1["value"] = ("HCHO", "C2H5OH", "C3H8","n-C4H10","n-C5H12","n-C6H14","C8H10","C4H9ClS","C3H9O3P","C4H8Cl2S","C7H16FO2P") ## The contents of a drop-down menu
canvas1.create_window(620,140, window=cm1)
cm1.bind('<<ComboboxSelected>>', calc) ## Binding 'calc' events
  
## Main interface for input of nine descriptor values (a single molecule diffusivity)
def values():       
    global New_HVF #our 1st input variable    
    New_HVF = float(entry1.get()) 
    
    global New_PLD #our 2nd input variable
    New_PLD = float(entry2.get()) 
    
    global New_LCD #our 2nd input variable
    New_LCD = float(entry3.get()) 
    
    global New_Density #our 2nd input variable
    New_Density = float(entry4.get()) 
    
    global New_VSA #our 2nd input variable
    New_VSA =float(entry5.get()) 
    
    global New_Dia #our 2nd input variable
    New_Dia = float(entry6.get()) 
    
    global New_Pol #our 2nd input variable
    New_Pol = float(entry7.get()) 
    
    global New_Dip #our 2nd input variable
    New_Dip = float(entry8.get())
    
    global New_Tb #our 2nd input variable
    New_Tb = float(entry9.get())
    
    global New_Tc #our 2nd input variable
    New_Tc = float(entry10.get())
    
    global New_Pc #our 2nd input variable
    New_Pc = float(entry11.get())

## LGBM Algorithm (The predictions of a single molecule diffusivity)   
    lgN = model.predict([[New_LCD, New_HVF, New_VSA, New_PLD, New_Density, New_Dia,
                          New_Pol, New_Dip, New_Tb,New_Tc, New_Pc]])    
    ## D transformation            
    N = pow(10,lgN)
    N1 = float(N)
    N2=format(N1,'.2E')
    N3= N2.split('E')  
    if (N3[1])[0] == "-":
        N3= N3[0]+" x 10^"+ N3[1].lstrip('0')
    else:
        N3=N3[0]+" x 10^"+ (N3[1])[1:].lstrip('0')    
    
    ## label of the predicted result
    Prediction_result  = (N3)   
    label_Prediction = tk.Label(root, font=('microsoft yahei',10),width=16,height=2,
                                text= Prediction_result)
    canvas1.create_window(270, 380, window=label_Prediction)

    ## D label
    '''
    lbo1=tk.Label(root, font=('microsoft yahei',12,"italic"),
                                text='N:')
    canvas1.create_window(100, 380, window=lbo1)
    '''
    
    ## unit label
    lbo2=tk.Label(root, font=('microsoft yahei',10),
                                text='(mol/kg)')
    canvas1.create_window(270, 405, window=lbo2) 

## button to call the 'values' command above       
button1 = tk.Button (root,font=('microsoft yahei',10), text='Predicted N',command=values) 
button1.configure(bg='orange')
canvas1.create_window(130, 380, window=button1)

## Batch prediction of material diffusivity
label_Z1 = tk.Label(root,font=('microsoft yahei',12),text='Batch prediction of material adsorption')
canvas1.create_window(415, 470, window=label_Z1)

## Open File
def open_file():
    filename = filedialog.askopenfilename(title='open exce')
    entry_filename.delete(0,"end")
    entry_filename.insert('insert', filename)
 
button_import = tk.Button(root, text="Import File",font=('microsoft yahei',10),command=open_file)
button_import.configure(bg='orange')
canvas1.create_window(280, 520, window=button_import)
 
## Import File
entry_filename = tk.Entry(root,font=('microsoft yahei',10),width=30)
canvas1.create_window(520, 520, window=entry_filename)

## Output LGBM model prediction results
def print_file():

    ## get extract contents of entry
    a = entry_filename.get() 

    ## Load the dataset
    pred_data1=pd.read_excel(a)
 
    ## Divide the data set
    pred_data=pred_data1.dropna(axis=0)

    ## Divide the data set
    df = pd.DataFrame(pred_data,columns=[ 'LCD', 'HVF', 'VSA', 'PLD', 'Density', 'Dia',
         'Pol', 'Dip', 'Tb ','Tc','Pc','lgN'])
    X_pred = df[['LCD', 'HVF', 'VSA', 'PLD', 'Density', 'Dia',
         'Pol', 'Dip', 'Tb ','Tc','Pc']].astype(float)

    ## Standardization
    transfer=StandardScaler()
    X_pred=transfer.fit_transform(X_pred)  
  
    ##model prediction
    Y_predict2 = model.predict(X_pred) 
    N = pow(10,Y_predict2) ## D transformation

    ## output result
    d1 = pd.DataFrame({'N_pred':N}) 
    newdata = pd.concat([pred_data,d1],axis=1) 
    newdata.to_excel("../Result/Batch_Predicted_N.xlsx")
    
    ## label_P (Prediction complete)
    label_P = tk.Label(root, font=('microsoft yahei',12),
                                text='Predicted results have default stored in:\nResult/Batch_Predicted_N.xlsx', bg='green')
    canvas1.create_window(450, 600, window=label_P)    
    
## Prediction button
but_pre=tk.Button(root,font=('microsoft yahei',10)
             , text='Batch Predicted N', bg='orange', command=print_file)
but_pre.configure(bg='orange')
canvas1.create_window(160, 600, window=but_pre)

root.mainloop() 

