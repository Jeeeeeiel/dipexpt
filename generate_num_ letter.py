# /usr/bin/python3
# __*__ coding: utf-8 __*__


from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np


# im = Image.open('/Users/Jeiel/Desktop/tmp/12_10.bmp')
# im = im.resize((18,18))
# data = np.array(im)
# print(data)


ht = ImageFont.truetype('simhei.ttf', size=20)
for i in range(10):
    im = Image.new('L', (18, 18), 255)
    draw = ImageDraw.Draw(im, mode='L')
    draw.text((4, -2), str(i), font=ht, fill='#000000')
    im.save('/Users/Jeiel/Desktop/ht_20_number&eng/' + str(i) + '.jpg')


for i in range(65, 91):
    im = Image.new('L', (18, 18), 255)
    draw = ImageDraw.Draw(im, mode='L')
    draw.text((4, -2), chr(i), font=ht, fill='#000000')
    im.save('/Users/Jeiel/Desktop/ht_20_number&eng/' + chr(i) + '_upper.jpg')


for i in range(97, 123):
    im = Image.new('L', (18, 18), 255)
    draw = ImageDraw.Draw(im, mode='L')
    draw.text((4, -2), chr(i), font=ht, fill='#000000')
    im.save('/Users/Jeiel/Desktop/ht_20_number&eng/' + chr(i) + '_lower.jpg')
