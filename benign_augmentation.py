import pickle
import numpy as np
from PIL import Image
import random

np.set_printoptions(threshold=np.nan)

benigns = pickle.load(open("/home/lgy1425/p1/total/benign_train.txt"))['x']

extension = []

for b in benigns :

    horizontal_flip = np.fliplr(np.array(b))

    b70 = np.array(b).astype(float)*float(0.7)

    x = random.randrange(5,15)
    y = random.randrange(5,15)

    cropped = np.array(b)[x:x+30,y:y+30]
    cropped = Image.fromarray(np.uint8(b*255))
    zoom = cropped.resize((50,50))
    zoom = np.asarray(zoom, dtype=np.float16) / float(223)

    i = Image.fromarray(np.uint8(b*255))

    t30 = i.rotate(30)

    for y in range(t30.size[1] ):
        for x in range(t30.size[0] ):
            color = t30.getpixel((x, y))
            if color == 0:
                t30.putpixel((x, y), random.randrange(1,11))

    tilt30 = np.asarray(t30,dtype=np.float16) / float(223)

    t45 = i.rotate(45)

    for y in range(t45.size[1]):
        for x in range(t45.size[0]):
            color = t45.getpixel((x, y))
            if color == 0:
                t45.putpixel((x, y), random.randrange(1, 11))

    tilt45 = np.asarray(t45, dtype=np.float16) / float(223)

    #extension.append(horizontal_flip)
    extension.append(b70)
    extension.append(b130)
    #extension.append(zoom)
    extension.append(tilt20)
    extension.append(tilt40)
    extension.append(tilt_30)

y_arr = np.full((len(extension),3),[0,1,0])

result = {"x":np.array(extension,dtype=np.float16),"y":y_arr}
print np.array(extension).shape,y_arr.shape

pickle.dump(result,open("/home/lgy1425/p1/total/benign_extension4.txt",'w'))





