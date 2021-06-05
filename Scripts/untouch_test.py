import os
import time


class PrintDeletedFiels():
    def __init__(self,dir_a,dir_b) -> None:
        self.dir_a=dir_a
        self.dir_b=dir_b

    def print_deleted_fiels(self):
        if not os.path.isdir(self.dir_a) or not os.path.isdir(self.dir_b):
            print('{} or {} is not a correct dir'.format(self.dir_a,self.dir_b))
            return None
        name_set=set()
        for item in os.listdir(self.dir_b):
            name_set.add(item)
        total=0
        different=0
        f=open('E:\different_items.txt','w')
        for item in os.listdir(self.dir_a):
            total+=1
            if not item in name_set:
                f.write(item+'\n')
                print(item)
                different+=1
        print('Found {} items in {} and {} of them does not exist in {}'.format(total,self.dir_a,different,different,self.dir_b))


class Rename():
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir

    def rename(self):
        if not os.path.isdir(self.root_dir):
            print("%s is not a correct dir" % self.root_dir)
            return None
        i = 0
        for item in os.listdir(self.root_dir):
            if item.endswith('.jpg'):
                src = os.path.join(self.root_dir, item)
                dst = os.path.join(self.root_dir, str(i) + '.jpg')
                os.rename(src, dst)
                print("Rename {} to {} successfully".format(item, str(i) + '.jpg'))
                i += 1


class BackUpFiels():
    def __init__(self,source_dir,target_dir) -> None:
        self.source_dir=source_dir
        self.target_dir=target_dir

    def BackUpFiels(self):
        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)
        target=os.path.join(self.target_dir,time.strftime('%Y%m%d')+'.zip')
        zip_command='winrar a {} {}'.format(target,' '.join(self.source_dir))
        print(zip_command)
        print('Back Up Starting:')
        if os.system(zip_command)==0:
            print('Back Up Fiels To {} Successfully'.format(target))
        else:
            print('Back Up Failed')

