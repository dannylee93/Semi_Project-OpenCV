import os
import shutil

rm_files = []
first_rm = []


move_path = './data/'
if not os.path.exists(move_path):
    os.mkdir(move_path)

first_path = './baza_slika/'
first_list = sorted(os.listdir(first_path))

for a in first_list :
     if a == '.DS_Store' or a[-3:] == 'txt' :
         first_rm.append(a)
         #print (first)
for b in first_rm :
    DS_path = os.path.join(first_path, b)
    print (DS_path)
    os.remove(DS_path)

for i in first_list :
    second_path = os.path.join(first_path, i)
    photo_list = sorted(os.listdir(second_path))
    photo_path = os.path.join(second_path, i)

    for j in photo_list:
        if j[-3:] == ".db":
            rm_files.append(j)
            rm_path = os.path.join(second_path,j)
            print ("deleting useless files")
            os.remove(rm_path)
        photo_path = os.path.join(second_path, j)
        shutil.copy(photo_path, os.path.join(move_path, j))
        print ("copy done", len(j))
