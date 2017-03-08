import pymongo
import dicom
import numpy as np
import pickle

test_set = ['089','090','091','092','093']

def crop(roi_dir, dcm_dir, wh) :
    roi_dicom = dicom.read_file(roi_dir)
    roi_img_arr = roi_dicom.pixel_array
    i = -1
    roi_coordinate = []
    roi_count = 0

    x_list = []
    y_list = []

    for ria in roi_img_arr :
        i += 1
        j = -1
        for r in ria :

            j += 1

            if r > 0 :
                roi_coordinate.append((i,j))
                x_list.append(i)
                y_list.append(j)
                roi_count += 1

    height = max(x_list) - min(x_list)
    width  = max(y_list) - min(y_list)

    ct_dicom = dicom.read_file(dcm_dir)
    ct_img_arr_origin = ct_dicom.pixel_array

    have_roi_list = []
    not_have_roi_list = []

    for x in range(0,ct_img_arr_origin.shape[0]-51,10) :
        for y in range(0,ct_img_arr_origin.shape[1]-51,10) :
            overlay_count = 0

            for w in range(x,x+wh+1) :
                for h in range(y,y+wh+1) :
                    if (w,h) in roi_coordinate :
                        overlay_count += 1
            if height > 50 and width > 50 :
                isroi = overlay_count / float(roi_count) > 0.4
            else :
                isroi = overlay_count / float(roi_count) > 0.7


            ct_img_arr = np.array(ct_img_arr_origin,dtype="float16")
            cropped_ct_img_arr = ct_img_arr[x:x+wh,y:y+wh] / float(1000)
            if isroi :
                have_roi_list.append(cropped_ct_img_arr)
            else :
                if x % 20 != 0 and y % 20 != 0 :
                    not_have_roi_list.append(cropped_ct_img_arr)


    return have_roi_list,not_have_roi_list


connection = pymongo.MongoClient("mongodb://localhost:25321/")
mongo_data = connection.data
dir = mongo_data.ndir
cropped50 = mongo_data.cropped50
data = dir.find({"is_roi":1,"bm":"malignant"})

x_have = []
y_have = []

x_have_test =[]
x_not_test = []

x_not = []
y_not = []

i=0
for d in data :
    i += 1

    if i <= 790 :
        continue

    print i
    have_roi_list, not_have_roi_list = crop(d['roi_dir'],d['ct_dir'],50)

    for h in have_roi_list :
        # cropped50.insert({"x":h,"y":[0,1,0]})

        if d['patient'][1:] in test_set :
            x_have_test.append(h)
        else :
            x_have.append(h)



    for n in not_have_roi_list:

        if d['patient'][1:] in test_set:
            x_not_test.append(h)
        else:
            x_not.append(n)

    if i % 10 == 0:
        result_x = np.array(x_have)
        result_test = np.array(x_have_test)

        pickle.dump(result_x, open("/home/lgy1425/p1/total/split_malignant_roi"+str(i)+".txt", 'w'))

        if len(result_test) > 0:
            pickle.dump(result_test, open("/home/lgy1425/p1/total/split_malignant_roi_test"+str(i)+".txt", 'w'))

        result_x = np.array(x_not)
        result_test = np.array(x_not_test)
        pickle.dump(result_x, open("/home/lgy1425/p1/total/split_malignant_not_roi"+str(i)+".txt", 'w'))
        if len(result_test) > 0:
            pickle.dump(result_test, open("/home/lgy1425/p1/total/split_malignant_not_roi_test"+str(i)+".txt", 'w'))

        x_have = []

        x_have_test = []
        x_not_test = []

        x_not = []





    if i == data.count():
        result_x = np.array(x_have)
        result_test = np.array(x_have_test)

        pickle.dump(result_x, open("/home/lgy1425/p1/total/split_malignant_roi" + str(i) + ".txt", 'w'))

        if len(result_test) > 0:
            pickle.dump(result_test, open("/home/lgy1425/p1/total/split_malignant_roi_test" + str(i) + ".txt", 'w'))

        result_x = np.array(x_not)
        result_test = np.array(x_not_test)
        pickle.dump(result_x, open("/home/lgy1425/p1/total/split_malignant_not_roi" + str(i) + ".txt", 'w'))
        if len(result_test) > 0:
            pickle.dump(result_test, open("/home/lgy1425/p1/total/split_malignant_not_roi_test" + str(i) + ".txt", 'w'))

        x_have = []

        x_have_test = []
        x_not_test = []

        x_not = []
