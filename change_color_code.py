#reference:IM = preprocess_warpAffine(img) https://amroamroamro.github.io/mexopencv/matlab/cv.invertAffineTransform.html
#reference:preprocess_warpAffine https://blog.csdn.net/u010420283/article/details/124033200?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522CED8CE97-0775-4599-B659-D899481FB648%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=CED8CE97-0775-4599-B659-D899481FB648&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-124033200-null-null.142^v100^pc_search_result_base9&utm_term=IM%20%3D%20cv2.invertAffineTransform%28M%29&spm=1018.2226.3001.4187
#reference:postprocess https://doc.openfoam.com/2306/tools/post-processing/utilities/postProcess/
#reference:IoU https://blog.csdn.net/briblue/article/details/91366128
#reference: NMs https://blog.csdn.net/weixin_44302770/article/details/134539912?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522D819AFFB-7FA4-4037-8788-FF0E412A9A67%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=D819AFFB-7FA4-4037-8788-FF0E412A9A67&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-134539912-null-null.142^v100^pc_search_result_base9&utm_term=NMS&spm=1018.2226.3001.4187
#reference: crop_mask https://github.com/nasaharvest/crop-mask
#refrence: process mask https://healpix.sourceforge.io/html/fac_process_mask.htm
#refrence:hsv2bgr https://stackoverflow.com/questions/35472650/hsv2bgr-conversion-fails-in-python-opencv-script
#reference: from ultralytics.nn.autobackend import AutoBackend https://docs.ultralytics.com/reference/nn/autobackend/
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend


def preprocess_warpAffine(image, dst_width=640, dst_height=640):
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    ox = (dst_width - scale * image.shape[1]) / 2
    oy = (dst_height - scale * image.shape[0]) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)

    img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    IM = cv2.invertAffineTransform(M)

    img_pre = (img_pre[..., ::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre, IM


def iou(box1, box2):
    def area_box(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    left, top = max(box1[:2], box2[:2])
    right, bottom = min(box1[2:4], box2[2:4])
    union = max((right - left), 0) * max((bottom - top), 0)
    cross = area_box(box1) + area_box(box2) - union
    if cross == 0 or union == 0:
        return 0
    return union / cross


def NMS(boxes, iou_thres):
    remove_flags = [False] * len(boxes)

    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue

        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if remove_flags[j]:
                continue

            jbox = boxes[j]
            if (ibox[5] != jbox[5]):
                continue
            if iou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes


def postprocess(pred, conf_thres=0.25, iou_thres=0.45):
    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        label = item[4:-32].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        left = cx - w * 0.5
        top = cy - h * 0.5
        right = cx + w * 0.5
        bottom = cy + h * 0.5
        boxes.append([left, top, right, bottom, confidence, label, *item[-32:]])

    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    return NMS(boxes, iou_thres)


def crop_mask(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    c, mh, mw = protos.shape 
    ih, iw = shape
    masks = (masks_in.float() @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw) 

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes) 
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0] 
    return masks.gt_(0.5)


def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)


def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

names={0: 'null', 1: 'accessories', 2: 'bag', 3: 'belt', 4: 'blazer', 5: 'blouse', 7: 'boots',
       8: 'bra', 9: 'bracelet', 10: 'cape', 12: 'clogs', 13: 'coat', 14: 'dress', 15: 'earrings'
       , 17: 'glasses', 19: 'hair', 20: 'hat', 21: 'heels', 22: 'hoodie', 23: 'intimate',
       24: 'jacket', 25: 'jeans', 26: 'jumper', 27: 'leggings', 28: 'loafers', 29: 'necklace', 30: 'panties',
       31: 'pants', 33: 'purse', 35: 'romper', 36: 'sandals', 37: 'scarf', 38: 'shirt',
       39: 'shoes', 40: 'shorts', 41: 'skin', 42: 'skirt', 43: 'sneakers', 44: 'socks', 45: 'stockings', 46: 'suit',
       47: 'sunglasses', 48: 'sweater', 50: 'swimwear', 51: 't-shirt', 52: 'tie', 53: 'tights',
       54: 'top', 55: 'vest', 57: 'watch', 58: 'wedges'}

if __name__ == "__main__":
    img = cv2.imread("/content/dataset/images/test/0503.jpg")  
    inputName= 31 
    color = np.array(random_color(inputName))  #np.array([0,0,255])

    img_pre, IM = preprocess_warpAffine(img)

    model = AutoBackend(weights="/content/runs/segment/train4/weights/best.pt") 
    names = model.names
    result = model(img_pre)
    """
    result[0] -> 1, 116, 8400 -> det head
    result[1][0][0] -> 1, 144, 80, 80
    result[1][0][1] -> 1, 144, 40, 40
    result[1][0][2] -> 1, 144, 20, 20
    result[1][1] -> 1, 32, 8400
    result[1][2] -> 1, 32, 160, 160 -> seg head
    """

    output0 = result[0].transpose(-1, -2) 
    output1 = result[1][2][0] 

    pred = postprocess(output0)
    pred = torch.from_numpy(np.array(pred).reshape(-1, 38))

    # pred -> nx38 = [cx,cy,w,h,conf,label,32]
    masks = process_mask(output1, pred[:, 6:], pred[:, :4], img_pre.shape[2:], True)

    boxes = np.array(pred[:, :6])
    lr = boxes[:, [0, 2]]
    tb = boxes[:, [1, 3]]
    boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]
    boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]


    h, w = img.shape[:2]
    for i, mask in enumerate(masks):
        mask = mask.cpu().numpy().astype(np.uint8)  # 640x640
        mask_resized = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR)  # 1080x810
        label = int(boxes[i][5])
        if label==inputName:
            colored_mask = (np.ones((h, w, 3)) * color).astype(np.uint8)
            masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)
            mask_indices = mask_resized == 1
            img[mask_indices] = (img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)

    cv2.imwrite("infer-seg.jpg", img)
    print("save done")




