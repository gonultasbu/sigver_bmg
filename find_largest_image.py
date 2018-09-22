from PIL import Image
import os

sizes=[0,0]
for root, dirs, filenames in os.walk("/home/mert/Desktop/Deep_Learning_Databases/unpacked_datasets/GPDSSyntheticSignatures4k", topdown=False):
    for file in filenames:
        if (file.endswith(".jpg")):
            sizes_temp = [Image.open(os.path.join(root,file), 'r').size[0],Image.open(os.path.join(root,file), 'r').size[1]]
            if (sizes_temp[0]>sizes[0]):
                sizes[0]=sizes_temp[0]
            else:
                pass
            if (sizes_temp[1]>sizes[1]):
                sizes[1]=sizes_temp[1]
            else:
                pass
#For the correct ordering
print sizes[::-1]