import cv2
import mediapipe as mp
import numpy as np
import scipy.spatial
import random
from sklearn.cluster import KMeans
from tkinter import *
from tkinter import filedialog
import os
import time
import math

win = Tk()
win.title("Virtual Painter")
win.geometry("1000x600") # width x height

class ImageSegmentation():

    def __init__(self, model=1): # 0 for general & 1 for landscape (faster)

        self.model = model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(self.model)

    def removeBG(self, img, imgBg=(255, 255, 255), threshold=0.1): # param threshold: higher = more cut, lower = less cut

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.selfieSegmentation.process(self.imgRGB)
        
        # join all the numpy arrays of img along a new axis
        condition = np.stack( 
            (self.results.segmentation_mask,) * 3, axis=-1) > threshold 

        if isinstance(imgBg, tuple): # if no background given

            _imgBg = np.zeros(img.shape, dtype=np.uint8)
            _imgBg[:] = imgBg

            imgOut = np.where(condition, img, _imgBg) # returns the indices of elements in an input array where the given condition is satisfied

        else:
            imgOut = np.where(condition, img, imgBg) # condition : When True, yield x, otherwise yield y

        return imgOut

class detecthand():

    def __init__(self, mode = False, maxHands = 1, model_complexity = 1, detectionconf = 0.5, trackingconf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelcomplexity = model_complexity
        self.detectionconf = detectionconf
        self.trackingconf = trackingconf
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.maxHands,self.modelcomplexity,self.detectionconf,self.trackingconf)
        self.mpdraw=mp.solutions.drawing_utils
        self.drawstyles = mp.solutions.drawing_styles

    def findhands(self, img, draw=True):
        
        # img.flags.writeable = False
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # Process the frame and return the results
        # print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handlandmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, handlandmark, self.mphands.HAND_CONNECTIONS)
        return img

    def handposition(self, img, hands=0, draw=True):

        landmarks=[]

        if self.results.multi_hand_landmarks:
            myhands = self.results.multi_hand_landmarks[hands]

            for id, lm in enumerate(myhands.landmark):

                    height, width, channel = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    landmarks.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx,cy), 5, (63,63,220), cv2.FILLED)
                        cv2.circle(img, (cx,cy), 6, (255,255,255), 1)     

        return landmarks

def selecttool(variable, margin):

    if variable < 50 + margin: # variable < 50 + 170 = 220 because we have a margin of 150 pixels on the left side
        return "Line"

    elif variable < 100 + margin: # variable < 270
        return "Rectangle"

    elif variable < 150 + margin: # variable < 320
        return "Draw"

    elif variable < 200 + margin: # variable < 370
        return "Circle"

    elif variable < 250 + margin: # variable < 420
        return "Erase"
        
    else: # variable < 470
        return "Color"

def selectcolor(variable, margin):

    # color Format = # Blue, Green, Red
    if variable < 50 + margin: # variable < 50 + 170 = 220 because we have a margin of 150 pixels on the left side
        return "Sky Blue"

    elif variable < 100 + margin: # variable < 270
        return "Orange"

    elif variable < 150 + margin: # variable < 320
        return "Green"

    elif variable < 200 + margin: # variable < 370
        return "Brown"

    elif variable < 250 + margin: # variable < 420
        return "Red"

def draw(middle_mcpy, middle_tipy): # draw or not
    if (middle_mcpy - middle_tipy) > 50 : # if middle finger is raised along with index finger, then it will draw  
        return True
    else:
        return False

def paint():

    # labels and variables
    margin = 170 # left margin
    width = 300 + margin # 300 is width of tools image
    height = 50 # 50 is height of tools image
    ctool = "Select Tool" # for dispalying on the screen which tool is selected
    rad = 40 # radius of circle
    check = False # for initializing variables
    thick = 4 # thickness of line and circle and rectangle
    prex, prey = 0,0 # coordinates of previous point
    color = (0,0,0) # default color is purple
    dc = (0,255,255) # default color for drawing

    # Header
    listImg = os.listdir("Images")
    imgList = []
    for imgPath in listImg:
        img = cv2.imread(f'Images/{imgPath}')
        imgList.append(img)

        # print(imgPath)

    header = imgList[0]
    header = header.astype(np.uint8) # convert to unsigned integer 0-255
    # print(len(imgList))

    detector = detecthand(detectionconf = 0.85, trackingconf = 0.85)

    # White Canvas
    # white = cv2.imread("white.jpg")
    white = np.ones((480,640,3)) * 255 
    white = white.astype(np.uint8)

    # cv2.imshow("white", white)

    mask = np.ones((480,640)) * 255 
    mask = mask.astype(np.uint8)

    cap = cv2.VideoCapture(0) # (480, 640, 3) 480 is height, 640 is width, 3 is channels
    # cap.set(3, 1280)
    # cap.set(4, 720)

    if not cap.isOpened():
        raise IOError("Webcam cannot be opened.")

    while True:

        success, img = cap.read() # (480, 640, 3)
        img = cv2.flip(img, 1)
        # print(img.shape)

        # Find hands
        img = detector.findhands(img)
        landmarks = detector.handposition(img, draw=False)
        # print(landmarks)

        if len(landmarks) != 0:
                
                indextipx, indextipy = landmarks[8][1:] # index finger tip
                middletipx, middletipy = landmarks[12][1:] # middle finger tip
                pinkytipx, pinkytipy = landmarks[20][1:] # pinky finger tip
                middlemcpy = landmarks[9][2] # middle finger mcp
                indexmcpy = landmarks[5][2] # index finger mcp

                if ctool == "Color": # if pinky finger tip is on the tools image
                    header = imgList[6]
                    if pinkytipx > margin and pinkytipx < width and pinkytipy < height:
                        cv2.circle(img, (pinkytipx, pinkytipy), 20, (255,255,255), 2)

                        if not draw(indexmcpy, indextipy):
                                    label = selectcolor(pinkytipx, margin)

                                    if label == "Sky Blue":
                                        color = (239, 209, 56)
                                        ctool = "Select Tool"
                                        header = imgList[1]
                                    elif label == "Orange":
                                        color = (0, 165, 255)
                                        ctool = "Select Tool"
                                        header = imgList[2]
                                    elif label == "Green":
                                        color = (0, 132, 0)
                                        ctool = "Select Tool"
                                        header = imgList[3]
                                    elif label == "Brown":
                                        color = (0,32,71)
                                        ctool = "Select Tool"
                                        header = imgList[4]
                                    elif label == "Red":
                                        color = (63, 63, 255)
                                        ctool = "Select Tool"
                                        header = imgList[5]

                elif indextipx > margin and indextipx < width and indextipy < height: # if index finger tip is on the tools image 

                    cv2.circle(img, (indextipx, indextipy), 20, (255,255,255), 2) # draw a white circle on the index finger tip

                    if not draw(middlemcpy, middletipy):
                        if ctool != "Color":
                            ctool = selecttool(indextipx, margin) # select tool

                if ctool == "Line":
                    if draw(middlemcpy, middletipy):

                        if not check:
                            prex, prey = indextipx, indextipy
                            check = True

                        cv2.line(img, (prex, prey), (indextipx, indextipy), dc, thick)

                    else:
                        if check:
                            cv2.line(mask, (prex, prey), (indextipx, indextipy), 0, thick)
                            cv2.line(white, (prex, prey), (indextipx, indextipy), color, thick)
                            check = False

                if ctool == "Rectangle":
                    if draw(middlemcpy, middletipy):

                        if not check:
                            prex, prey = indextipx, indextipy
                            check = True

                        cv2.rectangle(img, (prex, prey), (indextipx, indextipy), dc, thick)

                    else:
                        if check:
                            cv2.rectangle(white, (prex, prey), (indextipx, indextipy), color, thick)
                            cv2.rectangle(mask, (prex, prey), (indextipx, indextipy), 0, thick)
                            check = False

                if ctool == "Draw":
                    if draw(middlemcpy, middletipy):
                        cv2.line(mask, (prex, prey), (indextipx, indextipy), 0, thick)
                        cv2.line(white, (prex, prey), (indextipx, indextipy), color, thick)
                        prex, prey = indextipx, indextipy

                    else:
                        prex, prey = indextipx, indextipy

                if ctool == "Circle":
                    if draw(middlemcpy, middletipy):

                        if not check:
                            prex, prey = indextipx, indextipy
                            check = True

                        cv2.circle(img, (prex, prey), int(np.sqrt((prex - indextipx)**2 + (prey - indextipy)**2)), dc, thick)

                    else:
                        if check:
                            cv2.circle(white, (prex, prey), int(np.sqrt((prex - indextipx)**2 + (prey - indextipy)**2)), color, thick)
                            cv2.circle(mask, (prex, prey), int(np.sqrt((prex - indextipx)**2 + (prey - indextipy)**2)), 0, thick)
                            check = False

                if ctool == "Erase":
                    if draw(middlemcpy, middletipy):
                        cv2.circle(img, (indextipx, indextipy), 30, (0,0,0), -1)
                        cv2.circle(mask, (indextipx, indextipy), 30, 250, -1)
                        cv2.circle(white, (indextipx, indextipy), 30, (255,255,255), -1)

        detector.results = cv2.bitwise_and(img, img, mask=mask)
        img[:,:,0] = detector.results[:,:,0]
        img[:,:,1] = detector.results[:,:,1]
        img[:,:,2] = detector.results[:,:,2]

        # Set header image
        img[:50, 170:470] = cv2.addWeighted(header, 0.6, img[:50, 170:470], 0.4, 0)
        cv2.putText(img, ctool, (margin + 312, 34), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('Virtual Painter', img)
        cv2.imshow("Canvas", white)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("painting.jpg", white)
            cap.release()
            cv2.destroyAllWindows()
            break

def removebg():

    segmentor = ImageSegmentation()

    win.filename = filedialog.askopenfilename(initialdir="/D:/DL/Tkinter",
    title="Select an Image", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("png files", "*.png")))
    pic = win.filename

    pic = cv2.imread(pic)
    pic = cv2.resize(pic, (640, 480))
    
    while True:
        
        cv2.imshow("Original", pic)

        if cv2.waitKey(1) & 0xFF == ord('r'):
            break

    outP = segmentor.removeBG(pic, threshold=0.8)

    while True:
        
        cv2.imshow("BG Removed", outP)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("removebg.jpg", outP)

        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def changebg():

    segmentor = ImageSegmentation()

    win.filename = filedialog.askopenfilename(initialdir="/D:/DL/Tkinter",
    title="Select an Image", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("png files", "*.png")))
    pic = win.filename

    pic = cv2.imread(pic)
    pic = cv2.resize(pic, (640, 480))

    while True:
        
        cv2.imshow("Original", pic)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    win.filename = filedialog.askopenfilename(initialdir="/D:/DL/Tkinter",
    title="Select BG Image", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("png files", "*.png")))
    bg = win.filename

    bg = cv2.imread(bg)
    bg = cv2.resize(bg, (640, 480))

    outP = segmentor.removeBG(pic, bg, threshold=0.8)

    while True:
        
        cv2.imshow("BG Changed", outP)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("bgchanged.jpg", outP)

        elif cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

def compute_color_probabilities(pixels, palette):
    distances = scipy.spatial.distance.cdist(pixels, palette)
    maxima = np.amax(distances, axis=1)

    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]
    return distances

def get_color_from_prob(probabilities, palette, rand=False):
    probs = np.argsort(probabilities)
    i = probs[-1]
    if rand:
        r = random.uniform(0,1)
        if r<=0.1: i = probs[random.randint(max(0, probabilities.shape[0] - 3), probabilities.shape[0] - 1)]
    return palette[i]

def randomized_grid(h, w, scale):
    assert (scale > 0)

    r = scale//2

    grid = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j

            grid.append((y % h, x % w))

    random.shuffle(grid)
    return grid

def limit_size(img, max_x, max_y=0):
    if max_x == 0:
        return img

    if max_y == 0:
        max_y = max_x

    ratio = min(1.0, float(max_x) / img.shape[1], float(max_y) / img.shape[0])

    if ratio != 1.0:
        shape = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        return cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    else:
        return img

def get_color_palette(img, n=10, scale=200):
    # scale down the image to speedup kmeans
    img = limit_size(img, scale)

    clt = KMeans(n_clusters=n)
    clt.fit(img.reshape(-1, 3))

    return clt.cluster_centers_

def complement(colors):
    return 255 - colors

def pointillist_art():
    
    start_time = time.time()
    rand = True
    colours = 10

    win.filename = filedialog.askopenfilename(initialdir="/D:/DL/Tkinter",
    title="Select an Image", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("png files", "*.png")))
    pic = win.filename
    
    img = cv2.imread(pic)
    img = cv2.resize(img, (640, 480))
    
    radius_width = int(math.ceil(max(img.shape) / 1000))
    print("Radius width: %d" % radius_width)

    palette = get_color_palette(img, colours)
    complements = complement(palette)
    palette = np.vstack((palette,complements))
    print("Color Palette Computed")
    
    print("Drawing image...")
    
    canvas = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    canvas[:,:] = (255,255,255)

    grid = randomized_grid(img.shape[0], img.shape[1], scale=3)

    pixel_colors = np.array([img[x[0], x[1]] for x in grid])
    
    color_probabilities = compute_color_probabilities(pixel_colors, palette)

    for i, (y, x) in enumerate(grid):
        color = get_color_from_prob(color_probabilities[i], palette, rand=rand)
        cv2.ellipse(canvas, (x, y), (radius_width, radius_width), 0, 0, 360, color, -1, cv2.LINE_AA)

    cv2.imshow("Pointlist Art", canvas)
    cv2.imwrite("point art.jpg", canvas)

    print ("Total Time taken: %d seconds" % round(time.time() - start_time, 2))

def watercolor():

    win.filename = filedialog.askopenfilename(initialdir="/D:/DL/Tkinter",
    title="Select an Image", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("png files", "*.png")))
    pic = win.filename

    img = cv2.imread(pic)
    img = cv2.resize(img, (640, 480))
    res = cv2.stylization(img, sigma_s=60, sigma_r=0.1)

    # sigma_s controls the size of the neighborhood. Range 1 - 200
    # sigma_r controls the how dissimilar colors within the neighborhood will be averaged. A larger sigma_r results in large regions of constant color. Range 0 - 1

    cv2.imshow("Water Color Art", res)
    cv2.imwrite("watercolor.jpg", res)

def sketch():

    win.filename = filedialog.askopenfilename(initialdir="/D:/DL/Tkinter",
    title="Select an Image", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("png files", "*.png")))
    pic = win.filename

    img = cv2.imread(pic)
    img = cv2.resize(img, (640, 480))
    res, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.02) # res is grey scale image, dst_color is color image

    # shade_factor ( range 0 to 0.1 ) is a simple scaling of the output image intensity. The higher the value, the brighter is the result.

    cv2.imshow("Pencil Sketch Art - B & W", res)
    cv2.imwrite("sketch.jpg", res)

    cv2.imshow("Pencil Sketch Art - Colored", dst_color)
    cv2.imwrite("sketchc.jpg", dst_color)

def enhance():

    win.filename = filedialog.askopenfilename(initialdir="/D:/DL/Tkinter",
    title="Select an Image", filetypes=(("jpg files", "*.jpg"), ("all files", "*.*"), ("png files", "*.png")))
    pic = win.filename

    img = cv2.imread(pic)
    img = cv2.resize(img, (640, 480))
    res = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    # sigma_s controls the size of the neighborhood. Range 1 - 200
    # sigma_r controls the how dissimilar colors within the neighborhood will be averaged. A larger sigma_r results in large regions of constant color. Range 0 - 1

    cv2.imshow("Details Art", res)
    cv2.imwrite("detailsenhance.jpg", res)

def art():
    pre_im = PhotoImage(file="b.png")

    can = Canvas(win, width=1000, height=600, bg="light blue")
    can.place(x=0, y=0)
    can.create_image(0, 0, image=pre_im, anchor="nw")

    can.create_text(499, 34, text="Virtual Artist", fill="black", font=("Comic Sans MS", 22))

    can.create_text(499, 95, text="Artistic Effects", fill="black", font=("Comic Sans MS", 37))

    startbutton = Button(win, text="Pointillist Art", font=("Comic Sans MS", 20), bg="black", fg="light grey", command = pointillist_art)
    # place button at centre
    startbutton.place(x=408, y=180)

    dietbutton = Button(win, text="Sketching Effect", font=("Comic Sans MS", 20), bg="black", fg="light grey",command = sketch)
    # place button at centre
    dietbutton.place(x=380, y= 255)

    abutton = Button(win, text="Watercolor Effect", font=("Comic Sans MS", 20), bg="black", fg="light grey", command = watercolor)
    # place button at centre
    abutton.place(x=371, y= 330)

    startbutton = Button(win, text="Enhance Details", font=("Comic Sans MS", 20), bg="black", fg="light grey", command = enhance)
    # place button at centre
    startbutton.place(x=392, y=405)

    qbutton = Button(win, text="Back", font=("Comic Sans MS", 20), bg="black", fg="light grey", width=5, command = startmenu)
    # place button at centre
    qbutton.place(x=461, y= 480)

    win.mainloop()

def startmenu():
    pre_im = PhotoImage(file="bga.png")

    can = Canvas(win, width=1000, height=600, bg="light blue")
    can.place(x=0, y=0)
    can.create_image(0, 0, image=pre_im, anchor="nw")

    can.create_text(499, 34, text="Nice to see you again!", fill="black", font=("Comic Sans MS", 22))

    can.create_text(500, 95, text="Virtual Artist", fill="black", font=("Comic Sans MS", 37))

    startbutton = Button(win, text="Let's Paint", font=("Comic Sans MS", 20), bg="black", fg="light grey", command = paint)
    # place button at centre
    startbutton.place(x=431, y=180)

    abutton = Button(win, text="Change Image BG", font=("Comic Sans MS", 20), bg="black", fg="light grey", command = changebg)
    # place button at centre
    abutton.place(x=386, y= 255)

    dietbutton = Button(win, text="Remove Image BG", font=("Comic Sans MS", 20), bg="black", fg="light grey",command = removebg)
    # place button at centre
    dietbutton.place(x=383, y= 330)

    startbutton = Button(win, text="Artistic Effects", font=("Comic Sans MS", 20), bg="black", fg="light grey", command = art)
    # place button at centre
    startbutton.place(x=392, y=405)

    qbutton = Button(win, text="Quit", font=("Comic Sans MS", 20), bg="black", fg="light grey", width=5, command = win.destroy)
    # place button at centre
    qbutton.place(x=461, y= 480)

    win.mainloop()


# driver code
startmenu()


