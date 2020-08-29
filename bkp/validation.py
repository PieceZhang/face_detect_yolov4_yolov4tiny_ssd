import numpy as np

boxes = [[221.38578796, 1.67022705, 269.17785645, 100.79472351], #
         [131.87513733, 108.00149536, 180.33787537, 205.32565308],  #
         [59.07345963, 5.45892906, 63.29879379, 19.86083794],
         [62.66349411, 50.7850914, 100.89524841, 116.79486084],  #
         [109.20089722, -131.67922974, 544.19824219, 805.29663086]]
boxes = np.asarray(boxes)
scores = np.asarray([1, 2, 3, 4, 5])
classes = np.asarray([0, 0, 0, 0, 0])

if __name__ == '__main__':
    dellist = []
    for idx1, value1 in enumerate(boxes):
        if value1[2] - value1[0] < 20 or value1[3] - value1[1] < 20:
            dellist.append(idx1)
            continue
        for value2 in value1:
            if value2 > 416 or value2 < 0:
                dellist.append(idx1)
                break
    boxes = np.delete(boxes, dellist, 0)
    scores = np.delete(scores, dellist)
    classes = np.delete(classes, dellist)
    del dellist
    print(boxes, scores, classes)
