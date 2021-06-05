import time
from untouch_test import PrintDeletedFiels, Rename, BackUpFiels

if __name__=="__main__":
    start = time.time()
    '''PrintDeletedFiels=PrintDeletedFiels(dir_a='E:\A',dir_b='E:\B')
    PrintDeletedFiels.print_deleted_fiels()'''
    '''rename = Rename(root_dir="E:\壁纸\明星")
    rename.rename()'''
    BackUpFiels=BackUpFiels(source_dir=['E:\A'],target_dir='E:')
    BackUpFiels.BackUpFiels()
    print("time cost=%f" % (time.time()-start))
