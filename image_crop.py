import os
import cv2
import tqdm
img_source_dir = "/mnt/sda/Datasets/chuxinning/RGB_IR_0730"

dest_source_list = "/mnt/sda/Datasets/chuxinning/RGB_IR_0730_crop"

train_modal = "train_A"
list_train_A = os.listdir(os.path.join(img_source_dir,train_modal))
for img in tqdm.tqdm(list_train_A):
    image = cv2.imread(os.path.join(img_source_dir,train_modal,img))
    image_crop = image[16:-48,36:-28,:] # [352,352,3]
    cv2.imwrite(os.path.join(dest_source_list,train_modal,img),image_crop)

train_modal = "train_B"
list_train_A = os.listdir(os.path.join(img_source_dir,train_modal))
for img in tqdm.tqdm(list_train_A):
    image = cv2.imread(os.path.join(img_source_dir,train_modal,img))
    image_crop = image[16:-48,36:-28,:] # [352,352,3]
    cv2.imwrite(os.path.join(dest_source_list,train_modal,img),image_crop)

train_modal = "test_A"
list_train_A = os.listdir(os.path.join(img_source_dir,train_modal))
for img in tqdm.tqdm(list_train_A):
    image = cv2.imread(os.path.join(img_source_dir,train_modal,img))
    image_crop = image[16:-48,36:-28,:] # [352,352,3]
    cv2.imwrite(os.path.join(dest_source_list,train_modal,img),image_crop)

train_modal = "test_B"
list_train_A = os.listdir(os.path.join(img_source_dir,train_modal))
for img in tqdm.tqdm(list_train_A):
    image = cv2.imread(os.path.join(img_source_dir,train_modal,img))
    image_crop = image[16:-48,36:-28,:] # [352,352,3]
    cv2.imwrite(os.path.join(dest_source_list,train_modal,img),image_crop)
