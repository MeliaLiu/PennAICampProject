import cv2
import numpy as np

def padding(img, size=(224,224)):
    h, w, c = img.shape
    temp = np.zeros((max(h,w), max(h,w), c), dtype=np.uint8) + 255
    if h >= w:
        margin = (h-w)//2
        temp[:,margin:margin+w,:] = img
    else:
        margin = (w-h)//2
        temp[margin:margin+h,:,:] = img

    return cv2.resize(temp, size)      


def preprocess(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  
    ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
    cv2.drawContours(image,contours,-1,(255,255,255),3) 

class Operation:

    def __init__(self, window_name, img, save_path):
        self.mouse_pressed = False
        self.mouse_loc_button_down = [0,0]
        self.mouse_loc_button_up = [0,0]
        self.select = False
        self.window_name = window_name
        self.img = img
        self.img_show = self.img.copy()
        self.save_path = save_path

    def refresh(self):
        cv2.imshow(self.window_name, self.img_show)
        if self.select:
            lu = self.mouse_loc_button_down ### left up corner
            rd = self.mouse_loc_button_up ### right down corner
            selected_region = self.img.copy()[lu[0]:rd[0], lu[1]:rd[1], :]

            cv2.imshow("Selected Region", selected_region)
            cv2.imwrite(self.save_path, selected_region)
            
def mouseCallback(event, x, y, flags, param):
    op = param

    if event == cv2.EVENT_LBUTTONDOWN and not op.select and not op.mouse_pressed:
        op.mouse_pressed = True
        op.mouse_loc_button_down = [y, x]

    elif event == cv2.EVENT_LBUTTONUP and not op.select and op.mouse_pressed:
        op.mouse_pressed = False
        op.mouse_loc_button_up = [y, x]
        op.select = True
        op.img_show = op.img.copy()

    elif event == cv2.EVENT_MOUSEMOVE and op.mouse_pressed:
        op.img_show = op.img.copy()
        cv2.rectangle(op.img_show, (op.mouse_loc_button_down[1], op.mouse_loc_button_down[0]), (x,y), (255,255,255), 1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        op.select = False
        op.mouse_loc_button_down = [0,0]
        op.mouse_loc_button_up = [0,0]
        op.mouse_pressed = False
        cv2.destroyWindow('Selected Region')
        # op.i += 1

    if op.select:
        op.img_show = op.img.copy()
        cv2.rectangle(op.img_show, (op.mouse_loc_button_down[1], op.mouse_loc_button_down[0]), (op.mouse_loc_button_up[1], op.mouse_loc_button_up[0]), (255,255,255), 1)

    op.refresh()


def select_char(img_path, save_path):
    window_name = 'test'
    img = cv2.imread(img_path)

    op = Operation(window_name, img, save_path)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouseCallback, op)

    while(True):
        op.refresh()
        if cv2.waitKey() == 27: ### ESC
            break
        elif cv2.waitKey() == 8 or cv2.waitKey() == 127: ### backspace or del
            op.select = False
            op.mouse_loc_button_down = [0,0]
            op.mouse_loc_button_up = [0,0]
            op.mouse_pressed = False
            cv2.destroyWindow('Selected Region')
            # op.i += 1

    cv2.destroyAllWindows()

