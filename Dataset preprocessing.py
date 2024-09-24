import scipy.io
import os
import cv2
import numpy as np

#reference：scipy.io.loadmat from this link:https://snyk.io/advisor/python/scipy/functions/scipy.io.loadmat
#                                          https://www.slingacademy.com/article/using-io-loadmat-function-in-scipy-4-examples/?utm_content=cmp-true
#reference:About cv2.imread from this link:https://pythonexamples.org/python-opencv-cv2-find-contours-in-image/
#reference:About np.unique from this link: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
#reference:About os.path.join from this link:https://www.pythonlore.com/manipulating-file-paths-with-os-path-join-in-python/
#About the LLM:I used chatgpt4 to help me fix the code when it reported an error
def methold(name):
    img = cv2.imread(os.path.join(imgPath, name+".jpg"))
    data = scipy.io.loadmat(os.path.join(matPath, name+".mat"))
    gt = data['groundtruth']
    res = np.unique(gt, return_index=False, return_counts=False, return_inverse=False)

    fp = open(os.path.join(labelTXt, name+".txt"), 'w')
    for i in range(1, len(res)):
        label = ((gt == res[i]) * 255.0).astype(np.uint8)
        contours, _ = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            temp = str(res[i])
            for num in range(cnt.shape[0]):
                xw = cnt[num][0][0] / label.shape[1]
                yh = cnt[num][0][1] / label.shape[0]
                temp += " " + str(xw)
                temp += " " + str(yh)
            fp.writelines(str(temp))
            fp.writelines('\n')
    cv2.imwrite(os.path.join(outImgpath, name+".jpg"),img)
    cv2.imwrite(os.path.join(outpath, name + ".jpg"), gt)

if __name__ == '__main__':
    matPath="\content\clothing-co-parsing-master\annotations\pixel-level"
    imgPath="\content\clothing-co-parsing-master\photos"
    labelTXt="\content\clothing-co-parsing-master\NewClothing\labels\train"
    outpath="\content\clothing-co-parsing-master\NewClothing\Annotations"
    outImgpath="\content\clothing-co-parsing-master\NewClothing\images\train"
    name='0001.mat'
    for img in os.listdir(matPath):
        print(img.split('.')[0])
        methold(img.split('.')[0])


