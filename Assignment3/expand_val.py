import os

for dir_name in os.listdir('bird_dataset2/val_images/'):
    nb_images_to_add = 8 - len(os.listdir('bird_dataset2/val_images/'+dir_name))
    if nb_images_to_add > 0:
        # print("dir "+dir_name+str(nb_images_to_add))
        files_to_move = os.listdir('bird_dataset2/train_images/'+dir_name)[:nb_images_to_add]
        for file in files_to_move:
            # print(file)
            os.system('cp bird_dataset2/train_images/'+dir_name+'/'+file+' bird_dataset2/val_images/'+dir_name)


