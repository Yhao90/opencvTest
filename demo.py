import cv2  # opencv的缩写为cv2
import matplotlib.pyplot as plt  # matplotlib库用于绘图展示
import numpy as np  # numpy数值计算工具包

img = cv2.imread('resource/01_cat.jpg')
print(img.shape)  # (414, 500, 3)   (h,w,c)或yxz c表示 3 通道，这个 3 通道被 opencv 读进来是 BGR 的先后顺序的 3 通道
# 读取灰度图
img_gray = cv2.imread('resource/01_cat.jpg', cv2.IMREAD_GRAYSCALE)
print('type(img_gray):', type(img_gray))
print('img_gray.size: ', img_gray.size)  # 414 × 500 = 20700
print('img_gray.dtype:', img_gray.dtype)
print('img_gray.shape:', img_gray.shape)  # 只有一个通道


# 函数 展示图片
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


# 展示图片
# cv_show('image_cat_gray', img_gray)
# 保存图片
# cv2.imwrite('02_cat_gray.jpg', img_gray)

# 选择图片感兴趣的区域
# cat = img[0:100, 0:200]
# cv_show('cat', cat)
#
# b, g, r = cv2.split(img)
# cv_show('cat_b', b)
# # B通道，单通道，灰度图
# print('b.shape:', b.shape)
# cv_show('cat_g', g)
# # G通道，单通道，灰度图
# print('g.shape:', g.shape)
# cv_show('cat_r', r)
# # R通道，单通道，灰度图
# print('r.shape:', r.shape)
# img = cv2.merge((b, g, r))
# # 3 通道，彩色图
# print('img.shape:', img.shape)

# 只保留 R
# b, g, r = cv2.split(img)
# img = cv2.merge((b, g, r))
# cur_img = img.copy()
# cur_img[:, :, 0] = 0
# cur_img[:, :, 1] = 0
# cv_show('R', cur_img)

# img[:, :, 0] = 0
# img[:, :, 2] = 0
# cv_show('G', img)

# cur_img[:, :, 1] = 0
# cur_img[:, :, 2] = 0
# cv_show('B', cur_img)
"""
边界填充，扩大原始图像的范围，对扩大的地方进行的操作
参数
- BORDER_REPLICATE：复制法，也就是复制最边缘像素。
- BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
- BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
- BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg
- BORDER_CONSTANT：常量法，常数值填充。
"""
"""
# 填充多少区域
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50) 

# 最后一个入口参数为填充方式

# 方式一：复制法
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)
# 方式二：反射法
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
# 方式三：反射法二(不要最边缘的像素)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
# 方式四：外包装法
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
# 方式五：常量法
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, 0)
# .subplot(231) 第一个代表行数，第二个代表列数，第三个代表索引位置。
# 举例：plt.subplot(2, 3, 5) 和plt.subplot(235) 一样。所有数字不能超过10。
plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REPLECT')
plt.subplot(234), plt.imshow(wrap, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTAVI')
plt.show()
"""
# 阈值越界处理 , 不同数据大小不能执行数值计算操作,如相加,需要resize
# img_cat2 = img_cat + 10 # 将 img_cat 矩阵中每一个值都加 10 ，0-255 若相加越界后 294 用 294%256 获得余数 38
# cv2.add(img_cat,img_cat2)[:5,0] # cv2.add 是越界后取最大值 255

# resize 图像调整大小,会变形 ,图像融合 addWeighted
img_dog = cv2.imread('resource/03_dog.jpg')
img_dog = cv2.resize(img_dog, (500, 414))
res = cv2.addWeighted(img, 0.4, img_dog, 0.6, 0)  # img_cat 的权重为 0.4，img_dog 的权重为 0.6
print(img_dog.shape)
# plt.imshow(res)
# plt.show()
# (0,0)表示不确定具体值，fx=3 相当于行像素 x 乘 3，fy=1 相当于 y 乘 1
res = cv2.resize(img, (0, 0), fx=3, fy=1)
# plt.imshow(res)
# plt.show()

"""
图像阈值

ret, dst = cv2.threshold(src, thresh, maxval, type)

- src： 输入图，只能输入单通道图像，通常来说为灰度图
- thresh： 阈值
- dst： 输出图
- ret： 阈值
- maxval： 当像素值超过了阈值 ( 或者小于阈值，根据 type 来决定 )，所赋予的值
- type：二值化操作的类型，包含以下5种类型： 
- cv2.THRESH_BINARY   超过阈值部分取maxval ( 最大值 )，否则取0
- cv2.THRESH_BINARY_INV  THRESH_BINARY的反转
- cv2.THRESH_TRUNC  大于阈值部分设为阈值，否则不变
- cv2.THRESH_TOZERO  大于阈值部分不改变，否则设为0
- cv2.THRESH_TOZERO_INV  THRESH_TOZERO的反转
"""
img_gray = cv2.imread('resource/01_cat.jpg', cv2.IMREAD_GRAYSCALE)
ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
print(ret)
# THRESH_BINARY_INV 相对 THRESH_BINARY 黑的变成白的，白的变成黑的
ret, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
print(ret)
ret, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
print(ret)
ret, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
print(ret)
ret, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
print(ret)

titles = ['original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()

'''
图像平滑处理
1、均值滤波 
简单的平均卷积操作，方框中的值相加，取平均，替换掉中心204的值
(3,3) 为核的大小，通常情况核都是奇数 3、5、7
blur = cv2.blur(img,(3,3)) 
cv2.imshow('blur',blur)
2. 方框滤波
基本和均值一样，可以选择归一化
在 Python 中 -1 表示自适应填充对应的值，这里的 -1 表示与颜色通道数自适应一样
box = cv2.boxFilter(img,-1,(3,3),normalize=True)  # 方框滤波如果做归一化，得到的结果和均值滤波一模一样
3. 高斯滤波
# 高斯函数，越接近均值时，它的概率越大。
# 离中心值越近的，它的权重越大，离中心值越远的，它的权重越小。
aussian = cv2.GaussianBlur(img,(5,5),1)
4. 中值滤波
# 排序后拿中值替代中间元素值的大小
median = cv2.medianBlur(img,5)
'''
blur = cv2.blur(img, (3, 3))
box = cv2.boxFilter(img, -1, (3, 3), normalize=True)  # 方框滤波如果做归一化，得到的结果和均值滤波一模一样
aussian = cv2.GaussianBlur(img, (5, 5), 1)
median = cv2.medianBlur(img, 5)
res = np.hstack((img, blur, aussian, median))  # 矩阵横着拼接
# res = np.vstack((blur,aussian,median)) # 矩阵竖着拼接
# cv_show("t", res)

'''
腐蚀，膨胀，梯度运算
# 腐蚀操作通常是拿二值图像做腐蚀操作
ones()函数返回给定形状和数据类型的新数组，其中元素的值设置为1。
kernel = np.ones((5,5),np.uint8)

只要(5,5)框里有黑色，中心点的值就变为黑色，即原来的白色被黑色腐蚀掉
erosion = cv2.erode(img,kernel,iterations=1)

膨胀
dilate_1 = cv2.dilate(img,kernel,iterations=1)

开运算: 先腐蚀，再膨胀,消除小黑点，毛刺
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel) 
闭运算: 先膨胀，再腐蚀，排除小黑洞
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel) 
梯度运算 腐蚀-膨胀 ，突出边缘，保留物体的边缘轮廓
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel) 
礼帽预算,原始带刺，开运算不带刺，原始输入-开运算 = 刺 ,突出比原轮廓亮的部分
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
黑帽运算,闭运算-原始输入 ，突出比原轮廓暗的部分
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)

'''
pie = cv2.imread('resource/06_pie.png')
kernel = np.ones((30, 30), np.uint8)
erosion_1 = cv2.erode(pie, kernel, iterations=1)
erosion_2 = cv2.erode(pie, kernel, iterations=2)
erosion_3 = cv2.erode(pie, kernel, iterations=3)
# res = np.hstack((erosion_1, erosion_2, erosion_3))
# cv_show("t", res)

'''
Sobel算子、Scharr算子与Laplacian算子
在梯度运算中，很可能会出现负的灰度值，因此不再使用unit8而使用cv2.CV_64F这种带负数范围的类型，运算结果里的负数也要处理，否则显示的时候就会
截断为0，这样会丢失边界信息，使用convertScaleAbs函数把负值变为正值

Sobel算子函数：cv2.Sobel(src, ddepth, dx, dy, ksize)，返回值为Sobel算子处理后的图像。
 - ddepth：图像的深度
 - dx 和 dy 分别表示水平和竖直方向
 - ksize 是 Sobel 算子的大小
靠近最近点的左右和上下的权重最高，所以为±2。

scharr与sobel算子思想一样，只是卷积核的系数不同
scharr更敏感

Laplacian算子用的是二阶导，对噪音点更敏感一些。如果中心点是边界，它与周围像素点差异的幅度会较大，Laplacian算子根据此特点可以把边界识别出来。
'''
pie = cv2.imread('resource/06_pie.png')  # 读取图像
# 白到黑是整数，黑到白是负数了，所有的负数会被截断成 0，所以要取绝对值
sobelx = cv2.Sobel(pie, cv2.CV_64F, 1, 0, ksize=3)  # 1,0 表示只算水平方向梯度
sobelx = cv2.convertScaleAbs(sobelx)
# cv_show('sobelx', sobelx)
sobely = cv2.Sobel(pie, cv2.CV_64F, 0, 1, ksize=3)  # 0,1 只算 y 方向梯度
sobely = cv2.convertScaleAbs(sobely)  # 取负数时，取绝对值
# cv_show('sobely', sobely)
# 计算 x 和 y 后，再求和,不建议直接用xy 1，1参数去计算，分别计算再相加更合适
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)  # 0是偏置项
# cv_show('sobelxy', sobelxy)
img = cv2.imread('resource/07_Lena.jpg', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
# cv_show('sobelxy',sobelxy)
# 不同算子的差异 ,
img = cv2.imread('resource/07_Lena.jpg', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)  # 没有 x、y，因为是求周围点的比较
laplacian = cv2.convertScaleAbs(laplacian)

res = np.hstack((sobelxy, scharrxy, laplacian))
# cv_show('res', res)

'''
Canny边缘检测流程
- 1) 使用高斯滤波器，以平滑图像，滤除噪声。
- 2) 计算图像中每个像素点的梯度强度和方向。
- 3) 应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
- 4) 应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。 越小越敏感
- 5) 通过抑制孤立的弱边缘最终完成边缘检测。
cv2.Canny(img,80,150) # 第二个参数为minVal，第三个参数为maxVal
'''
img = cv2.imread('resource/07_Lena.jpg', cv2.IMREAD_GRAYSCALE)

v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
# cv_show('res', res)
'''
图像金字塔
可以做图像特征提取，做特征提取时有时可能不光对原始输入做特征提取，可能还会对好几层图像金字塔做特征提取。
可能每一层特征提取的结果是不一样的，再把特征提取的结果总结在一起。

1.高斯金字塔
cv2.pyrDown(img)向下采样，缩小。将G与高斯内核卷积，将所有偶数行和列去除
cv2.pyrUp(img)  向上采样，放大。将图像在每个方向扩大为原来的两倍，新增的行和列以0填充，使用先前同样的内核乘4与放大的图像卷积，获取近似值
2.拉普拉斯金字塔
拉普拉斯金字塔的每一层图像尺寸不变。
每一层操作都是上一层处理后作为输入，该输入减去该输入缩小放大后的图像，获得该层的输出
'''
# 高斯金字塔
img = cv2.imread('resource/09_AM.png')
up = cv2.pyrUp(img)
# cv_show('up', up)
down = cv2.pyrDown(img)
# cv_show('down', down)
# 拉普拉斯金字塔
domn = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
L_1 = img - down_up
# cv_show('L_1', L_1)
'''
图像轮廓
轮廓检测
contours, hierarchy = cv2.findContours(img,mode,method)
绘制轮廓，参数：图像、轮廓、轮廓索引(-1是自适应，画所有轮廓)，颜色模式，线条厚度
cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
mode
RETR_EXTERNAL ：只检索最外面的轮廓。
RETR_LIST：检索所有的轮廓，并将其保存到一条链表当中。
RETR_CCOMP：检索所有的轮廓，并将他们组织为两层：顶层是各部分的外部边界，第二层是空洞的边界。
RETR_TREE：检索所有的轮廓，并重构嵌套轮廓的整个层次。( 最常用 )
method
CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓。所有其他方法输出多边形 ( 顶点的序列 )。
CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
'''
# 图像二值化
img = cv2.imread('resource/08_Car.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 返回的ret 是阈值，thresh是处理后的像素矩阵
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 大于 127 的取 255，小于 127 的取 0
# cv_show('thresh', thresh)
# 轮廓检测
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(np.array(hierarchy).shape)  # 轮廓点的信息
print(hierarchy)  # hierarchy 是把轮廓结果保存在层级结构当中，暂时用不上
# 绘制所有轮廓,参数：图像、轮廓、轮廓索引(-1是自适应，画所有轮廓)，颜色模式，线条厚度
# 若不用拷贝后的，而是用原图画轮廓，则画轮廓图绘把原始的输入图像重写，覆盖掉
draw_img = img.copy()
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
# cv_show('res', res)
cnt = contours[0]  # 通过轮廓索引，拿到该索引对应的轮廓特征
print(cv2.contourArea(cnt))  # 该轮廓的面积
print(cv2.arcLength(cnt, True))  # 该轮廓的周长，True表示闭合的
# 轮廓近似
img = cv2.imread('resource/11_contours2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 大于17的取255，小于127的取0
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
draw_img = img.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 0, 255), 2)
# cv_show('res', res)
epsilon = 0.1 * cv2.arcLength(cnt, True)  # 周长的百分比，这里用 0.1 的周长作阈值
approx = cv2.approxPolyDP(cnt, epsilon, True)  # 第二个参数为阈值
draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
# cv_show('res', res)
# 轮廓为外接矩形
img = cv2.imread('resource/10_contours.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 大于17的取255，小于127的取0
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[6]
x, y, w, h = cv2.boundingRect(cnt)  # 可以得到矩形四个坐标点的相关信息
# 画矩形，第二个参数是左上角坐标，第三个是右下脚坐标
img = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255), 2)
# cv_show('img', img)
# 画外接圆
draw_img = img.copy()
(x, y), redius = cv2.minEnclosingCircle(cnt)
center = (int(x), int(y))
redius = int(redius)
img = cv2.circle(draw_img, center, redius, (0, 255, 0), 2)
# cv_show('img', img)

'''
模板匹配
① 模板匹配和卷积原理很像，模板在原图像上从原点开始滑动，计算模板与（图像被模板覆盖的地方）的差别程度(例如值127与值190的区别)，
这个差别程度的计算方法在opencv里有6种，然后将每次计算的结果放入一个矩阵里，作为结果输出。
② 假如原图形是AxB大小，而模板是axb大小，则输出结果的矩阵是(A-a+1)x(B-b+1)。
③ 模板匹配计算方式6种方式 ( 用归一化后的方式更好一些 )：
TM_SQDIFF：计算平方不同，计算出来的值越小，越相关。
TM_CCORR：计算相关性，计算出来的值越大，越相关。
TM_CCOEFF：计算相关系数，计算出来的值越大，越相关。
TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关。
TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关。
TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关。
④ 公式：https://docs.opencv.org/3.3.1/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res) # 返回模板匹配后最小值、最大值及其位置，取大取小看第三个参数
'''
method = eval('cv2.TM_CCOEFF')
template = cv2.imread('resource/12_Face.jpg', 0)  # 0 表示以灰度图方式读取
img = cv2.imread('resource/13_Lena.jpg', 0)
h, w = template.shape[:2]  # 获得模板的宽和高
print(img.shape)
print(template.shape)
res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
print(res.shape)  # 返回的矩阵大小 (A-a+1)x(B-b+1)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # 返回模板匹配后最小值、最大值的位置
# 匹配多个
img_rgb = cv2.imread('resource/14_Mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
print('img_gray.shape：', img_gray.shape)
template = cv2.imread('resource/15_Mario_coin.jpg', 0)
print('template.shape：', template.shape)
h, w = template.shape[:2]
# res 是 result 的简称
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)  # res 是返回每一个小块窗口得到的结果值
threshold = 0.8

# 取匹配程度大于 80% 的坐标

loc = np.where(res >= threshold)  # np.where 使得返回 res 矩阵中值大于 0.8 的索引，即坐标
print('type(loc):', type(loc))  # loc 为元组类型
print('len(loc):', len(loc))  # loc 元组有两个值
print('len(loc[0]):', len(loc[0]), 'len(loc[1]):', len(loc[1]))  # loc 元组每个值 120 个元素
print('type(loc[0]):', type(loc[0]), 'type(loc[1]):', type(loc[1]))  # loc 元组每个值的类型为 numpy.array
print("loc[::-1]：", loc[::-1])  # loc[::-1] 表示顺序取反，即第二个 numpy.array 放在第一个 numpy.array 前面

i = 0
# zip函数为打包为元组的列表，例 a = [1,2,3] b = [4,5,6] zip(a,b) 为 [(1, 4), (2, 5), (3, 6)]
for pt in zip(*loc[::-1]):  # 当用 *b 作为传入参数时, b 可以为列表、元组、集合，zip使得元组中两个 numpy.array 进行配对
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)
    i = i + 1
print('i:', i)  # 120
# cv_show('img_rgb', img_rgb)

'''
角点检测
harris角点检测函数：cv2.cornerHarris()
img：数据类型为 ﬂoat32 的入图像。
blockSize：角点检测中指定区域的大小。
ksize：Sobel求导中使用的窗口大小。常用 3。
k：取值参数为 [0,04,0.06]。常用 0.04。
'''
img = cv2.imread('resource/17_Chessboard.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('res.shape:', gray.shape)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)  # 每个点与对应点的相似性地值，即变化值
print('dst.shape:', dst.shape)
img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 比相似性最大值的百分之一要大，则标注为角点
# cv_show('dst', img)

'''
sift(尺度不变特征变换)特征点检测
在一定的范围内，无论物体是大还是小，人眼都可以分辨出来，然而计算机要有相同的能力却很难，所以要让机器能够对物体在不同尺度下有一个统一的认知，
就需要考虑图像在不同的尺度下都存在的特点
高斯差分金字塔 (DOG) 差分结果较大的被视为比较重要的特征。
特征匹配

'''
img = cv2.imread('resource/18_House.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT.create()  # 将 SIFT 算法实例化出来
kp = sift.detect(gray, None)  # 把灰度图传进去，得到特征点、关键点
img = cv2.drawKeypoints(gray, kp, img)
# cv_show("sift", img)
kp, des = sift.compute(gray, kp)  # 计算特征点
print(np.array(kp).shape)  # 6809 个关键点
print(des.shape)  # 每个关键点有 128 维向量
print(des[0])  # 获得第 0 号关键点的值
# 特征匹配
img1 = cv2.imread('resource/19_Box.png', 0)
img2 = cv2.imread('resource/20_Box_in_scene.png', 0)
kp1, des1 = sift.detectAndCompute(img1, None)  # None是掩模
kp2, des2 = sift.detectAndCompute(img2, None)
# crossCheck 表示两个特征点要互相匹配，例如 A 中的第 i 个特征点与 B 中第 j 个特征点最近的，并且 B 中第 j 个特征点到 A 中的第 i 个特征点也是最近的。
# 将两幅图像的特征点、特征向量算出来，用欧氏距离去比较特征向量相似性，一般情况下默认用的是归一化后的欧式距离去做，为了使得结果更均衡些。
# 如果不用 sift 特征计算方法去做，而是用其他特征计算方法需要考虑不同的匹配方式,默认NORM_L2。
# normType：如 NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2.
#             NORM_L1 和 NORM_L2 更适用于 SIFT 和 SURF 描述子;
#             NORM_HAMMING 和 ORB、BRISK、BRIEF 一起使用；
#             NORM_HAMMING2 用于 WTA_K==3或4 的 ORB 描述子.
bf = cv2.BFMatcher(crossCheck=True)  # cv2.BFMatcher 蛮力匹配缩写
# 然后是1对1的匹配
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)  # 画出匹配结果前十个点
# cv_show('img3', img3)
# k对最佳匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)  # k 参数可选，可以一个点跟它最近的k个点可选
good = []
for m, n in matches:
    # m.distance 与 n.distance 比值小于 0.75，这是自己设定的过滤条件
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv_show('img3', img3)
# 使用FLANN

# FLANN，Fast Library for Approximate Nearest Neighbors. 其是针对大规模高维数据集进行快速最近邻搜索的优化算法库.
# FLANN Matcher 需要设定两个字典参数，以指定算法和对应的参数，分别为 IndexParams 和 SearchParams.
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
# cv_show('img3', img3)
# 使用随机抽样一致算法 (RANSAC) 过滤掉匹配不对的点,至少要有四对点。方式具体看图像拼接stich
