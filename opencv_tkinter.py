from PIL import ImageTk, Image
import cv2
import tkinter as tk
import tkinter.messagebox
from tkinter import Toplevel, filedialog
import matplotlib.pyplot as plt
import numpy as np
global drawing
changeSpeed = 200
#開檔
def open_file():
    filepath = filedialog.askopenfilename()
    if filepath != '':
        global img
        img_cv = cv2.imread(filepath)
        b ,g ,r = cv2.split(img_cv)
        tk_img = cv2.merge((r,g,b))
        img = Image.fromarray(tk_img)
        img = ImageTk.PhotoImage(img)
     #確認要開啟的圖片 是否成功開啟
    if cv2.haveImageReader(filepath) == True:     
        global tempimg
        tempimg = img_cv.copy()
        global panel
        panel = tk.Label(root, image = img)
        panel.image = img
        panel.place(x = 0 , y = 0)
        global h , w
        h = img.height()
        w = img.width()
        if h >500 or w > 500 :
            h1  = h + 30
            root.geometry(f"{w}x{h1}")
            tk_window = tk.Label(root, text='之後會做看看滑鼠座標偵測', font=('Arial', 12))
            tk_window.place(x = 0 , y = h + 5)
    else:
    #利用tkinter完成跳出提醒視窗
        tkinter.messagebox.showerror('錯誤','路徑不能有中文')
        
#寫檔
def write_file():
            #可自由存想要的路徑 但必須是英文
    filepath_write = filedialog.asksaveasfilename(filetypes=[('JPG', '.jpg')])
    if not filepath_write:
        return
    cv2.imwrite(filepath_write + '.jpg' , tempimg)
#RGB色彩空間
def img_RGB():
    global tempimg
    img_rgb = cv2.cvtColor(tempimg, cv2.COLOR_GRAY2BGR)
    b ,g ,r = cv2.split(img_rgb)
    cv_img = cv2.merge((r,g,b))
    re_rgb = Image.fromarray(cv_img)
    tempimg = img_rgb
    img = ImageTk.PhotoImage(re_rgb)
    panel = tk.Label(root, image = img)
    panel.image = img
    panel.place(x = 0 , y = 0)
#新建視窗用來調整hsv的滑輪
def hsv_windows():
        global low_h , low_s , low_v , high_h , high_s, high_v
        global color_space_window
    #建立hsv slider的視窗
        color_space_window = Toplevel()
        color_space_window.title("color_space_windows")
        color_space_window.geometry("1100x400")
        color_space_window.resizable(1,1) 
    #建立slider h最大值為180
        low_h = tk.Scale(color_space_window, from_=0, to=179,tickinterval=10, length=1000, orient="horizontal")
        low_s = tk.Scale(color_space_window, from_=0, to=255,tickinterval=10, length=1000, orient="horizontal")
        low_v = tk.Scale(color_space_window, from_=0, to=255,tickinterval=10, length=1000, orient="horizontal") 
    
        high_h = tk.Scale(color_space_window, from_=0, to=179,tickinterval=10, length=1000, orient="horizontal")
        high_s =tk.Scale(color_space_window, from_=0, to=255,tickinterval=10, length=1000, orient="horizontal")
        high_v =tk.Scale(color_space_window, from_=0, to=255,tickinterval=10, length=1000, orient="horizontal")
    #設定初始值(原圖)
        high_h.set(179)
        high_s.set(255)
        high_v.set(255)
    #自動調整slider的位址
        low_h.pack()
        low_v.pack()
        low_s.pack()
        high_h.pack()
        high_s.pack()
        high_v.pack()

        lowh = tk.Label(color_space_window , text= 'low_h')
        lowh.place(x = 10 , y = 20)
        lows = tk.Label(color_space_window , text= 'low_s')
        lows.place(x = 10 , y = 80)
        lowv = tk.Label(color_space_window , text= 'low_v')
        lowv.place(x = 10 , y = 140)
        highh = tk.Label(color_space_window , text= 'high_h')
        highh.place(x = 10 , y = 200)
        highs = tk.Label(color_space_window , text= 'high_s')
        highs.place(x = 10 , y = 260)
        highv = tk.Label(color_space_window , text= 'high_v')
        highv.place(x = 10 , y = 320)
        hsv_colorspace()
#hsv調整
def hsv_colorspace():
    #獲取slider的值
    hMin = low_h.get()
    hMax = high_h.get()
    sMin = low_s.get()
    sMax = high_s.get()
    vMin = low_v.get()
    vMax = high_v.get()

    lower = np.array([hMin, sMin, vMin])#設置過濾的顏色低值
    upper = np.array([hMax, sMax, vMax])#設置過濾的顏色高值
    hsv = cv2.cvtColor(tempimg, cv2.COLOR_BGR2HSV) #將圖片轉成hsv
    mask = cv2.inRange(hsv, lower, upper)#調節圖片颜色信息（H）、飽和度（S）、亮度（V)區間
    global output
    output = cv2.bitwise_and(tempimg,tempimg, mask= mask)
    cv2.imshow('color_space' , output)
    color_space_window.bind("<Escape>", stop_window) #偵測keyboard 為 esc時跳出迴圈
    root.after(changeSpeed,hsv_colorspace) #利用after來形成迴圈效果
    #跳出迴圈
def stop_window(event):
    b ,g ,r = cv2.split(output)
    tk_img = cv2.merge((r,g,b))
    re_hsv=Image.fromarray(tk_img)
    img =ImageTk.PhotoImage(re_hsv)
    panel = tk.Label(root, image = img)
    panel.image = img
    panel.place(x = 0 , y = 0)
    global tempimg
    tempimg = output
    color_space_window.focus_set()
    root.after_cancel(hsv_colorspace)
    color_space_window.destroy()
    cv2.destroyWindow('color_space')
    #灰階色彩
def img_gray():
    global tempimg , gray_img
    cv_gray = cv2.cvtColor(tempimg, cv2.COLOR_BGR2GRAY)
    tempimg = cv_gray
    gray_img = cv_gray
    tk_gray=Image.fromarray(cv_gray)
    img = ImageTk.PhotoImage(tk_gray)
    panel = tk.Label(root, image = img)
    panel.image = img
    panel.place(x = 0 , y = 0)
#彩色直方圖
def img_histograms():
        color = ('b','g','r')
        #依序設定b,g,r的值
        for i, col in enumerate(color):
            histr = cv2.calcHist([tempimg],[i],None,[256],[0, 256])# 計算直方圖每個 bin 的數值
            plt.plot(histr, color = col)#將b,g,r匯入
            plt.xlim([0, 256])#設定x軸 0~256
        plt.show()
#灰階直方圖
def gray_histograms():
    plt.hist(tempimg.ravel(), 256, [0, 256])
    plt.show()
def img_size():
    tkinter.messagebox.showinfo("照片的大小為 : " , (w ,"x", h))
    #灰階高通濾波
def Fourier_high_pass_test():
    global tempimg
    Fourier = gray_img
    img_float = np.float32(Fourier)
    dft = cv2.dft(img_float,flags=cv2.DFT_COMPLEX_OUTPUT)#傅立葉變換
    dft_shift = np.fft.fftshift(dft)#獲得頻譜圖，將低頻值轉換到中間
    rows,cols = Fourier.shape  #分別保存圖像的高和寬
    crow,col = int(rows/2), int(cols/2)#計算中心點坐標
    #構造高通濾波器，設置的越大，低頻信息刪除的越多
    mask = np.ones((rows,cols,2),np.uint8)
    mask[crow-10:crow+10, col-10:col+10] = 0 #以頻率為0處坐標為中心，寬10+10，高10+10的部分抹除
    #傅立葉逆變換
    fshift = dft_shift*mask #刪除中間的信息，保留其他部分的信息，低頻都集中在中央位置，統一刪除
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    #顯示圖片，比較原圖和處理過的圖
    plt.subplot(121), plt.imshow(Fourier, cmap='gray')
    plt.title('input img'), plt.xticks([]), plt.yticks([])#不顯示坐標軸
    tempimg = img_back
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('fft img'), plt.xticks([]), plt.yticks([])
    plt.show()
    #灰階低通濾波
def low_pass_filtering_test():
    global tempimg
    Fourier = gray_img
    img_float = np.float32(Fourier)#將圖片轉成np.float32類型
    dft = cv2.dft(img_float,flags=cv2.DFT_COMPLEX_OUTPUT)#傅立葉變換   
    dft_shift = np.fft.fftshift(dft)#獲得頻譜圖，將低頻值轉換到中間
    #獲取頻率為0部分中心點位置
    rows,cols = Fourier.shape #分別保存圖像的高和寬
    crow,col = int(rows/2), int(cols/2) #計算中心點坐標
    #構造低通濾波器
    mask = np.zeros((rows,cols,2),np.uint8)#構造的size和原圖相同2通道，傅立葉變換後有實部和虛部
    mask[crow-30:crow+30, col-30:col+30] = 255# 構造一個以頻率為0點中心對稱，長30+30，寬30+30的一個區域，只保留區域內部的頻率
    #傅立葉逆變換
    fshift = dft_shift*mask #頻譜圖上，低頻的信息都在中間，濾波器和頻譜圖相乘，遮擋四周，保留中間，中間是低頻
    f_ishift = np.fft.ifftshift(fshift)#在獲得頻譜圖時，將低頻點從邊緣點移動到圖像中間，現在要逆變換，得還回去
    img_back = cv2.idft(f_ishift)#傅立葉逆變換idft
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])#還原後的還是有實部和虛部，需要進一步處理
    #顯示圖片，比較原圖和處理過的圖
    plt.subplot(121), plt.imshow(Fourier, cmap='gray')
    plt.title('input img'), plt.xticks([]), plt.yticks([])#不顯示坐標軸
    tempimg = img_back
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('fft img'), plt.xticks([]), plt.yticks([])
    plt.show()
#彩色低通濾波 但感覺怪怪的 沒放進去程式裡
def low_pass_filtering():
    global tempimg
    B, G, R = cv2.split(tempimg)
    zeros = np.zeros(tempimg.shape[:2], dtype="float32")
    for offset in range(10, 101, 10):
        res_b = Fourier_low_pass(B, offset)
    # cv2.imwrite("res_b.jpg", cv2.merge([res_b, zeros, zeros]))
        res_g = Fourier_low_pass(G, offset)
    # cv2.imwrite("res_g.jpg", cv2.merge([zeros, res_g, zeros]))
        res_r = Fourier_low_pass(R, offset)
    # cv2.imwrite("res_r.jpg", cv2.merge([zeros, zeros, res_r]))
        res_merge = cv2.merge([res_b, res_g, res_r])
    cv2.imshow("low_pass_filtering" , res_merge)
def Fourier_low_pass(img, offset):
    #傅里叶变换
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    #设置低通滤波器
    rows, cols = img.shape
    crow,ccol = int(rows/2), int(cols/2) #中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-offset:crow+offset, ccol-offset:ccol+offset] = 1

    #掩膜图像和频谱图像乘积
    f = fshift * mask
    # print(f.shape, fshift.shape, mask.shape)

    #傅里叶逆变换
    ishift = np.fft.ifftshift(f)
    iimg = cv2.idft(ishift)
    # 频谱图像双通道复数转换为0-255区间
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    res = 255 * (res - np.min(res)) / (np.max(res) - np.min(res))
    return res
#彩色高通濾波 但感覺怪怪的 沒放進去程式裡
def high_pass_filtering():
    global tempimg
    B, G, R = cv2.split(tempimg)
    zeros = np.zeros(tempimg.shape[:2], dtype="float32")
    for offset in range(10, 101, 10):
        res_b = Fourier_high_pass(B, offset)
    # cv2.imwrite("res_b.jpg", cv2.merge([res_b, zeros, zeros]))
        res_g = Fourier_high_pass(G, offset)
    # cv2.imwrite("res_g.jpg", cv2.merge([zeros, res_g, zeros]))
        res_r = Fourier_high_pass(R, offset)
    # cv2.imwrite("res_r.jpg", cv2.merge([zeros, zeros, res_r]))
        res_merge = cv2.merge([res_b, res_g, res_r])
        '''
        B, G, R = cv2.split(res_merge)
        res_merge_tk = cv2.merge([R, G, B])
        tk_img = Image.fromarray(res_merge_tk)
        img =ImageTk.PhotoImage(tk_img)
        panel = tk.Label(root, image = img)
        panel.image = img
        panel.place(x = 0 , y = 0)
        '''
        cv2.imshow("high_pass_filtering" , res_merge)        
def Fourier_high_pass(img, offset):
    #傅里叶变换
    # dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    # fshift = np.fft.fftshift(dft)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    #设置高通滤波器
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - offset:crow + offset, ccol - offset:ccol + offset] = 0

    #傅里叶逆变换
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)
    # 频谱图像双通道复数转换为0-255区间
    # res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    res = iimg
    res = 255 * (res - np.min(res)) / (np.max(res) - np.min(res))
    return res
#圖片平移
def Panning():
    global tempimg
    panning_left_right = tkinter.simpledialog.askinteger(title = '獲取資訊',prompt='請輸入左右平移多少 ， 左請輸入負值',initialvalue = '0')
    panning_up_down = tkinter.simpledialog.askinteger(title = '獲取資訊',prompt='請輸入上下平移多少 ， 上請輸入負值',initialvalue = '0')
    b ,g ,r = cv2.split(tempimg)
    tk_img = cv2.merge((r,g,b))
    img = Image.fromarray(tk_img)
    img = ImageTk.PhotoImage(img)
    global panel
    panel.destroy()
    panel = tk.Label(root, image = img)
    panel.image = img
    panel.place(x = panning_left_right , y = panning_up_down)#直接更改坐標軸
#圖片旋轉
def Rotary():
    global tempimg
    Rotary = tkinter.simpledialog.askinteger(title = '獲取資訊',prompt='請輸入旋轉角度' , initialvalue = '0')
    rotated = rotate(tempimg, Rotary)
    tempimg = rotated
    cv2.imshow('rotate' , rotated)
def rotate(image, angle, center=None, scale=1.0):
    #獲取圖片大小
    (h, w) = image.shape[:2]
    #沒有指定座標中心，則將圖片中心設為旋轉中心
    if center is None:
        center = (w / 2, h / 2)
    #執行旋轉
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    #返回旋轉後的圖
    return rotated
#仿射轉換
def affine_transform():
    global tempimg
    rows, cols, ch = tempimg.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])#獲取原圖的三個點
    point1_col = tkinter.simpledialog.askinteger(title = '獲取資訊',prompt='請輸入第一個點的寬，請輸入1 ~ 10',initialvalue = '0')
    point1_row = tkinter.simpledialog.askinteger(title = '獲取資訊',prompt='請輸入第一個點的高，請輸入1 ~ 10',initialvalue = '0')
    point2_col = tkinter.simpledialog.askinteger(title = '獲取資訊',prompt='請輸入第二個點的寬，請輸入1 ~ 10',initialvalue = '0')
    point2_row = tkinter.simpledialog.askinteger(title = '獲取資訊',prompt='請輸入第二個點的高，請輸入1 ~ 10',initialvalue = '0')  
    point3_col = tkinter.simpledialog.askinteger(title = '獲取資訊',prompt='請輸入第三個點的寬，請輸入1 ~ 10',initialvalue = '0')
    point3_row = tkinter.simpledialog.askinteger(title = '獲取資訊',prompt='請輸入第三個點的高，請輸入1 ~ 10',initialvalue = '0')      
    #映射後的三個座標值
    pts2 = np.float32([[cols * (point1_col / 10)  , rows * (point1_row / 10)], [cols * (point2_col / 10), rows * (point2_row / 10)], [cols * (point3_col / 10), rows * (point3_row / 10)]])
    #由三個點對計算變換矩陣 
    M = cv2.getAffineTransform(pts1, pts2)
    #將原圖型和變換矩陣和圖片大小 進行仿射變換
    dst = cv2.warpAffine(tempimg, M, (cols, rows))
    tempimg = dst
    cv2.imshow('image', dst)
#透射轉換
def perspective_transform():
        global tempimg
        PT_img = tempimg
        pts1 = []
        rows, cols , ch = PT_img.shape#取的原圖長寬
        pts2 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1] , [cols-1 , rows - 1]])#原圖的4個座標點 順序為 左上 右上 左下 右下
        cv2.imshow("image" , PT_img)
        cv2.setMouseCallback("image", mouse, param = (PT_img, pts1))#取的滑鼠座標
        cv2.waitKey(0)#按下任意後再繼續 否則會直接執行下去
        cv2.destroyAllWindows()
        print("pts1:", pts1)
        pts1 = np.float32(pts1[:4]) #將4個座標點轉成no.float32
        #生成透視變換矩陣
        M = cv2.getPerspectiveTransform(pts1 , pts2) #由於是內建的函數 所以原圖和要更新的座標點順序都要一樣
        #進行透視變換
        dst = cv2.warpPerspective(PT_img, M, (PT_img.shape[1], PT_img.shape[0]))
        cv2.imshow('image', dst)
        tempimg = dst
#取得滑鼠座標
def mouse( event, x, y, flags, param):
    tempimg = param[0]
    pts1 = param[1]
        #取得按下左鍵的座標點
    if event == cv2.EVENT_LBUTTONDOWN:
        pts1.append([x, y])
        #將座標點用圓形顯示出來
        cv2.circle(tempimg, (x, y), 4, (0, 255, 255), thickness = -1)
        cv2.imshow("image", tempimg)
root = tk.Tk()
root.title('opencv_GUI')
#利用tk內建Menu來完成GUI
menubar = tk.Menu(root)
file = tk.Menu(menubar, tearoff = 0)
#建立下拉式Menu選單
menubar.add_cascade(label ='File', menu = file)
file.add_command( label ='open File', command = open_file)
file.add_command(label ='Save File', command = write_file)
file.add_separator()
file.add_command(label ='Exit', command = root.destroy)

color_space_Menu = tk.Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='色彩空間轉換', menu = color_space_Menu)
color_space_Menu.add_command( label ='RGB', command = img_RGB)
color_space_Menu.add_command(label ='HSV', command = hsv_windows)
color_space_Menu.add_command(label ='灰階', command = img_gray)

img_edit_Menu = tk.Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='影像資訊呈現', menu = img_edit_Menu)
img_edit_Menu.add_command( label ='彩色直方圖', command = img_histograms)
img_edit_Menu.add_command( label ='灰階直方圖', command = gray_histograms)
img_edit_Menu.add_command(label ='影像大小', command = img_size)

img_set_Menu = tk.Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='各種鄰域處理功能', menu = img_set_Menu)
img_set_Menu.add_command( label ='低通濾波器', command = low_pass_filtering_test)
img_set_Menu.add_command(label ='高通濾波器', command = Fourier_high_pass_test)

img_set_Menu = tk.Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='幾何轉換', menu = img_set_Menu)
img_set_Menu.add_command( label ='平移', command = Panning)
img_set_Menu.add_command(label ='旋轉', command = Rotary)
img_set_Menu.add_command( label ='仿射轉換', command = affine_transform)
img_set_Menu.add_command( label ='透射轉換', command = perspective_transform)
#將tk視窗預設500*500大小
root.geometry('500x500')
#可自由調整視窗大小
root.resizable(1, 1)

# 显示菜单
root.config(menu=menubar)

root.mainloop()