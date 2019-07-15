from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

# 読み込み処理
#print('入力してください')
#a = input()
#png_name = a
#png_name = request.files['img_file'].filename
png_name = 'test_n2'
model_file_name='16_ETL7-CNN-VGG-like_model'
model=load_model('./' + model_file_name+'.h5')
img_path = ('./' + png_name + '.png')

#読み込んだファイルの色を反転
imge = cv2.imread(img_path)
img_path2 = cv2.bitwise_not(imge)
cv2.imwrite('./image_sin.png',img_path2)
#パス名の更新
png_name = 'image_sin'
img_path = ('./' + png_name + '.png')

img = img_to_array(load_img(img_path, grayscale=True, target_size=(32, 32)))
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]

label=['あ', 'い', 'う', 'え', 'お',
            'か', 'き', 'く', 'け', 'こ', 
            'さ', 'し', 'す', 'せ', 'そ', 
            'た', 'ち', 'つ', 'て', 'と', 
            'な', 'に', 'ぬ', 'ね', 'の', 
            'は', 'ひ', 'ふ', 'へ', 'ほ', 
            'ま', 'み', 'む', 'め', 'も', 
            'や', 'ゆ', 'よ', 
            'ら', 'り', 'る', 'れ', 'ろ', 
            'わ', 'を', 'ん' ]

pred = model.predict(img_nad, batch_size=32, verbose=1)
score = np.max(pred)
len(pred)
pred_label = label[np.argmax(pred[0])]
print('name:',pred_label)
print('score:',score)
plt.imshow(img_path2, cmap='gray')