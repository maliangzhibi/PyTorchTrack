import cv2 as cv
from PIL import Image
import jpeg4py

def default_image_loader(path=None):
    
    im = jpeg4py_loader(path)
    if im is None:
        print('use cv_loder instead.')
        im = cv_loader(path)
        if im is None:
            print('use PIL_image_loader instead.')
            im = PIL_Image_loader(path)
            if im is None:
                raise Exception("loading image error.")   

    return im

def cv_loader(path=None):
    '''use opencv to read image
    args:
        path: the image placement path
    '''
    if path == None:
        raise Exception('the path in cv_loader is None, please check your setup.')

    try:
        im = cv.imread(path, cv.IMREAD_COLOR)
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('error: coudl not read image "{}"'%(path))
        print(e)
        return None

def jpeg4py_loader(path=None):
    '''
    use the jpeg4py for reading images
    args:
        path: the image placement path
    '''
    if path==None:
        raise Exception('the path in jpeg4py_loader is None, please check your setup.')

    try:
        im = jpeg4py.JPEG(path).decode()
        return im
    except Exception as e:
        print('error: could not read image "{}"' % (path))
        print(e)
        return None

def PIL_Image_loader(path=None):
    '''
    use the Image for reading images
    args:
        path: the image placement path
    '''
    if path==None:
        raise Exception('the path in PIL_Image_loader is None, please check your setup. ')
    try:
        im = Image.open()
        return im
    except Exception as e:
        print('error: could not read image "{}"' % (path))
        print(e)
        return None

if __name__ == "__main__":
    pass
