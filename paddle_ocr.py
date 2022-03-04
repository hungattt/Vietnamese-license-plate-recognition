from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = './Anh/crop01.jpg'
result = ocr.ocr(img_path, cls=True)
print(result)
print('Bien so xe la :',result[0][1][0],result[1][1][0])