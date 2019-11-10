from flask import Flask, jsonify, abort, make_response,render_template,url_for,request,redirect,send_file
import os.path
from flask_bootstrap import Bootstrap
import json
import secrets
# from imagenet import imagenet
from PIL import Image
import io
import qrcode as qr
import base64
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras.models import load_model
import json
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
bootstrap = Bootstrap(app)

app.config['SECRET_KEY'] = secrets.token_hex(16)
# よくわからない
# limit upload file size : 1MB という意味らしい
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['.png', '.jpg'])



@app.route("/")
def index():
    return render_template('index.html')


# ここから本命．画像のI/O処理
@app.route('/send', methods = ['POST'])
def posttest():
    png_name = request.files['img_file']
    result_name = "./" + png_name.filename # 最後のimg用に保存しておく
    # ファイルの拡張子を取得する
    _, ext = os.path.splitext(png_name.filename)
    # 小文字にして
    ext = ext.lower()
    # ランダムなファイル名を決める
    new_name = secrets.token_urlsafe(16) + ext.lower()
    # 縮小して保存
    i = Image.open(png_name)
    i.thumbnail((200, 200))
    i.save(os.path.join(UPLOAD_FOLDER, new_name))
    
    print(new_name)

    model_file_name = '16_ETL7-CNN-VGG-like_model' #モデル
    model = load_model('./' + model_file_name + '.h5')
    img_path = ('./' + new_name) # 一旦.png消した


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

    label_hiragana=['あ', 'い', 'う', 'え', 'お',
            'か', 'き', 'く', 'け', 'こ', 
            'さ', 'し', 'す', 'せ', 'そ', 
            'た', 'ち', 'つ', 'て', 'と', 
            'な', 'に', 'ぬ', 'ね', 'の', 
            'は', 'ひ', 'ふ', 'へ', 'ほ', 
            'ま', 'み', 'む', 'め', 'も', 
            'や', 'ゆ', 'よ', 
            'ら', 'り', 'る', 'れ', 'ろ', 
            'わ', 'を', 'ん' ]

    label = ['a', 'i', 'u', 'e', 'o',
            'ka', 'ki', 'ku', 'ke', 'ko', 
            'sa', 'shi', 'su', 'se', 'so', 
            'ta', 'chi', 'tsu', 'te', 'to', 
            'na', 'ni', 'nu', 'ne', 'no', 
            'ha', 'hi', 'hu', 'he', 'ho', 
            'ma', 'mi', 'mu', 'me', 'mo', 
            'ya', 'yu', 'yo', 
            'ra', 'ri', 'ru', 're', 'ro', 
            'wa', 'wo', 'n' ]

    pred = model.predict(img_nad, batch_size=32, verbose=1)
    score = np.max(pred)
    len(pred)
    pred_label = label_hiragana[np.argmax(pred[0])]
    print('name:',pred_label)
    print('score:',score)
    pred_label = str(pred_label)
    score = str(score)
    #plt.imshow(img_path2, cmap='gray')

    # 返す処理
    res_json = {
        'result': {
            'label': pred_label,
            'score': score,
        }
    }
    #return render_template('index.html', message=jsonify(res_json)) 
    return render_template('index.html',hiragana = "ひらがな:" + pred_label, score = "認識度:" + score, img_result = result_name, color = "red")
    #return jsonify(res_json)


# 結果ルーティング
@app.route('/kekka')
def kekka():
    return render_template('kekka.html')






# エラーハンドリング
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)
@app.errorhandler(413)
def oversize(error):
    return render_template('index.html',massege = "画像サイズが大きすぎます",color = "red")
@app.errorhandler(400)
def nosubmit(error):
    return render_template('index.html',massege = "画像を送信してください",color = "red")
@app.errorhandler(503)
def all_error_handler(error):
     return 'InternalServerError\n', 503
if __name__ == '__main__':
    app.run()
