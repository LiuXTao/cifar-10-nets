import os
my_dir = '/home/ubuntu/pyCase/Furniture-Classification-master/imaterialist-challenge-furniture-2018/train_data/'
id = 102377

while id <= 172618:
    my_file = "{}.jpg".format(id)
    id += 1
    if os.path.exists(my_dir+my_file):
        #删除文件，可使用以下两种方法。
        os.remove(my_dir + my_file)

        #os.unlink(my_file)
        print('remove: %s'%my_file)
    else:
        print("file %s not exist"%my_file)
print("finished!")