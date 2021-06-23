import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random
from skimage.transform import resize
import imageio
from tqdm import tqdm


def draw_single_char(ch, font, canvas_size=64, x_offset=0, y_offset=-13):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img


def resize_image(img, w=64, h=64):
    # pad to square
    pad_size = int(abs(img.shape[0] - img.shape[1]) / 2)
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
    # resize
    img = resize(img, (w, h))
    assert img.shape == (w, h)
    return img


def main(source_path, ratio, datafolder, charlist_path=None):

    random.seed(20171201)

    source_font = ImageFont.truetype(source_path, size=64)
    if not charlist_path:
        charlist = [chr(c) for c in np.random.choice([i for i in range(int('4E00', 16), int('9FFF', 16))], 6400, replace=False)]
    else:
        with open(charlist_path) as gf:
            glyphs = gf.read()
            charlist = list(set([x for x in glyphs if x not in ('、', '\n', ' ')]))
    sourcelist = []
    
    if not os.path.exists(datafolder):
        os.makedirs(datafolder)

    train_path = os.path.join(datafolder, 'train')
    test_path = os.path.join(datafolder, 'test')
    folders = [train_path, test_path]

    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)

    for ch in charlist:
        source_img = draw_single_char(ch, font=source_font)
        sourcelist.append(source_img)

    arr = np.arange(len(charlist))
    np.random.shuffle(arr)

    ntrain = np.floor(float(ratio) * len(charlist))

    for x in tqdm(np.arange(len(arr))):

        ch = charlist[arr[x]]
        source_img = sourcelist[arr[x]]

        if arr[x] <= ntrain:
            imageio.imwrite(os.path.join(
                train_path, str(ord(ch)) + '.png'), source_img)
        elif arr[x] <= ntrain:
            imageio.imwrite(os.path.join(
                train_path, str(ord(ch)) + '.png'), source_img)
        elif arr[x] > ntrain:
            imageio.imwrite(os.path.join(
                test_path, str(ord(ch)) + '.png'), source_img)
        else:
            imageio.imwrite(os.path.join(
                test_path, str(ord(ch)) + '.png'), source_img)


if __name__ == '__main__':
    main(
        source_path='/home/jscarlson/Downloads/sawarabi-mincho/sawarabi-mincho-medium.ttf', # '/home/jscarlson/Downloads/MPLUSRounded1c-Bold.ttf',
        datafolder='/home/jscarlson/Downloads/modern_src_font_imgs', # '/home/jscarlson/Downloads/modern_dst_font_imgs',
        charlist_path='/home/jscarlson/Documents/sawarabi_mincho_glyphs.txt', # '/home/jscarlson/Documents/jis_level_one_kanji.txt',
        ratio=0.9, 
    )
