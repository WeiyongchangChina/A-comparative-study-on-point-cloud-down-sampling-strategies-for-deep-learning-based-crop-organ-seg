'''
This code is transform .txt to the .pcd.
and this is the original point cloud
2019.5.23
'''
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
ROOT_DIR = os.path.join(ROOT_DIR, 'models/ASIS/out(4class)')

colour_map = [[0.592,0.153,0.698],[0.2666,0.733,0.706],[0.192,0.51,1.00],[0.305,0.71,0.263],
              [0.65, 0.95, 0.05], [0.65, 0.35, 0.65], [0.95, 0.65, 0.05],[0.35, 0.05, 0.05], 
              [0.65, 0.05, 0.05], [0.65, 0.35, 0.95], [0.05, 0.05, 0.65], 
              [0.65, 0.05, 0.35], [0.05, 0.35, 0.35], [0.65, 0.65, 0.35], 
              [0.35, 0.95, 0.05], [0.05, 0.35, 0.65], [0.95, 0.95, 0.35], 
              [0.65, 0.65, 0.65], [0.95, 0.95, 0.05], [0.65, 0.35, 0.05], 
              [0.35, 0.65, 0.05], [0.95, 0.65, 0.95], [0.95, 0.35, 0.65], 
              [0.05, 0.65, 0.95], [0.65, 0.95, 0.65], [0.95, 0.35, 0.95], 
              [0.05, 0.05, 0.95], [0.65, 0.05, 0.95], [0.65, 0.05, 0.65], 
              [0.35, 0.35, 0.95], [0.95, 0.95, 0.95], [0.05, 0.05, 0.05], 
              [0.05, 0.35, 0.95], [0.65, 0.95, 0.95], [0.95, 0.05, 0.05], 
              [0.35, 0.95, 0.35], [0.05, 0.35, 0.05], [0.05, 0.65, 0.35], 
              [0.05, 0.95, 0.05], [0.95, 0.65, 0.65], [0.35, 0.95, 0.95], 
              [0.05, 0.95, 0.35], [0.95, 0.35, 0.05], [0.65, 0.35, 0.35], 
              [0.35, 0.95, 0.65], [0.35, 0.35, 0.65], [0.65, 0.95, 0.35], 
              [0.05, 0.95, 0.65], [0.65, 0.65, 0.95], [0.35, 0.05, 0.95], 
              [0.35, 0.65, 0.95], [0.35, 0.05, 0.65], [1.00, 0.00, 0.00], 
              [0.00, 1.00, 0.00], [0.35, 0.05, 0.35], [0.95, 0.95, 0.65]]
    
def creat_pred_pcd(Full_Data, output_path):
    if os.path.exists(output_path):
        os.remove(output_path)
    Output_Data = open(output_path, 'a')
    # headers
    Output_Data.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgba\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1')
    string = '\nWIDTH ' + str(Full_Data.shape[0])
    Output_Data.write(string)
    Output_Data.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(Full_Data.shape[0])
    Output_Data.write(string)
    Output_Data.write('\nDATA ascii')
    # pack RGB
    for j in range(Full_Data.shape[0]):
        index = Full_Data[j,4]
        R=int(colour_map[int(index)][0]*255)
        G=int(colour_map[int(index)][1]*255)
        B=int(colour_map[int(index)][2]*255)
        value = (int(R) << 16 | int(G) << 8 | int(B))
        string = ('\n' + str(Full_Data[j,0]) + ' ' + str(Full_Data[j, 1]) + ' ' +str(Full_Data[j, 2]) + ' ' + str(value))
        Output_Data.write(string)
    Output_Data.close()
    
def creat_gt_pcd(Full_Data, Label_Data, output_path):
    # creat output file
    if os.path.exists(output_path):
        os.remove(output_path)
    Output_Data = open(output_path, 'a')
    # headers
    Output_Data.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgba\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1')
    string = '\nWIDTH ' + str(Full_Data.shape[0])
    Output_Data.write(string)
    Output_Data.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(Full_Data.shape[0])
    Output_Data.write(string)
    Output_Data.write('\nDATA ascii')
    # pack RGB
    for j in range(Full_Data.shape[0]):
        index = Label_Data[j,0]
        R=int(colour_map[int(index)][0]*255)
        G=int(colour_map[int(index)][1]*255)
        B=int(colour_map[int(index)][2]*255)
        value = (int(R) << 16 | int(G) << 8 | int(B))
        string = ('\n' + str(Full_Data[j,0]) + ' ' + str(Full_Data[j, 1]) + ' ' +str(Full_Data[j, 2]) + ' ' + str(value))
        Output_Data.write(string)
    Output_Data.close()
    
def creat_ins_pcd(Full_Data, output_path):
    # creat output file
    if os.path.exists(output_path):
        os.remove(output_path)
    Output_Data = open(output_path, 'a')
    # headers
    Output_Data.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgba\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1')
    string = '\nWIDTH ' + str(Full_Data.shape[0])
    Output_Data.write(string)
    Output_Data.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(Full_Data.shape[0])
    Output_Data.write(string)
    Output_Data.write('\nDATA ascii')
    # pack RGB
    for j in range(Full_Data.shape[0]):
        index = Full_Data[j,5]
        R=int(colour_map[int(index)][0]*255)
        G=int(colour_map[int(index)][1]*255)
        B=int(colour_map[int(index)][2]*255)
        value = (int(R) << 16 | int(G) << 8 | int(B))
        string = ('\n' + str(Full_Data[j,0]) + ' ' + str(Full_Data[j, 1]) + ' ' +str(Full_Data[j, 2]) + ' ' + str(value))
        Output_Data.write(string)
    Output_Data.close()
    
def creat_ins_gt_pcd(Full_Data, Label_Data, output_path):
    # creat output file
    if os.path.exists(output_path):
        os.remove(output_path)
    Output_Data = open(output_path, 'a')
    # headers
    Output_Data.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgba\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1')
    string = '\nWIDTH ' + str(Full_Data.shape[0])
    Output_Data.write(string)
    Output_Data.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(Full_Data.shape[0])
    Output_Data.write(string)
    Output_Data.write('\nDATA ascii')
    # pack RGB
    for j in range(Full_Data.shape[0]):
        index = Label_Data[j,1]
        R=int(colour_map[int(index)][0]*255)
        G=int(colour_map[int(index)][1]*255)
        B=int(colour_map[int(index)][2]*255)
        value = (int(R) << 16 | int(G) << 8 | int(B))
        string = ('\n' + str(Full_Data[j,0]) + ' ' + str(Full_Data[j, 1]) + ' ' +str(Full_Data[j, 2]) + ' ' + str(value))
        Output_Data.write(string)
    Output_Data.close()
    
if __name__=='__main__': 
    NUM_POINTS = 4096
    OUTPUT_PATH_LIST = [os.path.join(ROOT_DIR,line.rstrip()) for line in open(os.path.join(ROOT_DIR, 'output_filelist.txt'))]
    for i in range(len(OUTPUT_PATH_LIST)):
        
        input_data_path = OUTPUT_PATH_LIST[i]
        input_label_path = (OUTPUT_PATH_LIST[i])[:-8] + 'gt.txt'
        Full_Data_all = np.loadtxt(input_data_path)
        Label_Data_all = np.loadtxt(input_label_path)
        
        data_num = Full_Data_all.shape[0]
        plant_num = int(data_num/NUM_POINTS)
        
        for j in range (plant_num):
            print('Processing: %d/%d'%(j,plant_num))
            Full_Data = Full_Data_all[j*NUM_POINTS:(j+1)*NUM_POINTS,...]
            Label_Data = Label_Data_all[j*NUM_POINTS:(j+1)*NUM_POINTS,...]
            
            output_pred_path = os.path.join(ROOT_DIR,'PCD_real', 'plant_'+ str(j) + '_sem.pcd')
            output_gt_path = os.path.join(ROOT_DIR,'PCD_real',  'plant_'+ str(j) + '_sem_gt.pcd')
            output_ins_path = os.path.join(ROOT_DIR,'PCD_real',  'plant_'+ str(j) + '_ins.pcd')
            output_ins_gt_path = os.path.join(ROOT_DIR,'PCD_real',  'plant_'+ str(j) + '_ins_gt.pcd')
            
            creat_pred_pcd(Full_Data, output_pred_path)
            creat_gt_pcd(Full_Data, Label_Data, output_gt_path)
            creat_ins_pcd(Full_Data, output_ins_path)
            creat_ins_gt_pcd(Full_Data, Label_Data, output_ins_gt_path)
            
