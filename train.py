import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


# model.load('/root/ultralytics-main/yolov8s-seg.pt') # loading pretrain weights

# reference:Training parameters official detailed link：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings
# /ultralytics-main/ultralytics/yolov8-p6.yaml reference:https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p6.yaml
# /ultralytics-main/ultralytics/yolov8n-bifpn.yaml reference:https://blog.csdn.net/jisuaijicainiao/article/details/137468547?ops_request_misc=%257B%2522request%255Fid%2522%253A%252236D9E9B7-0275-40ED-86A9-701A75DEBAA9%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=36D9E9B7-0275-40ED-86A9-701A75DEBAA9&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-137468547-null-null.142^v100^pc_search_result_base9&utm_term=yolov8%E6%94%B9%E8%BF%9Bbifpn&spm=1018.2226.3001.4187
#/ultralytics-main/ultralytics/yolov8n-CBAM.yaml reference:https://blog.csdn.net/pyscl01/article/details/132693835?ops_request_misc=&request_id=&biz_id=102&utm_term=yolov8cbam%E6%80%8E%E4%B9%88%E5%8A%A0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-132693835.142^v100^pc_search_result_base9&spm=1018.2226.3001.4187

if __name__ == '__main__':
    """

    """
    model = YOLO('/content//ultralytics/yolov8-p6.yaml')
    model.train(data='/content/NewClothing.yaml',
                cache=False,
                imgsz=640,
                epochs=150,
                batch=8,  
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='auto', # 
                # patience=0,
                # resume=True, 
                # amp=False, 
                project='runs/train',
                name='raw',
                )