# Single-Cell-Feature-Extract

### 细胞图像的特征提取

---

#### 输入

1. 在./Data文件夹内放入原始图片(不限后缀)

#### 输出

1. 输出数据特征到./Out/Data.csv中

2. 输出轮廓绘制图片到./Out/中

3. 处理完后移动图片至./Finishi/中
   
   样例（部分）
   
   | filename | length           | area    | inscribedCircle | circumscribedCircle | specificValue     | rectangleDegree  | circleDegree      |
   | -------- | ---------------- | ------- | --------------- | ------------------- | ----------------- | ---------------- | ----------------- |
   | 1-40033  | 872.163554668426 | 43250.5 | 156             | 318                 | 0.490566037735849 | 0.82947889825871 | 0.714504761114803 |

4. 输出特征提取参照图片到./Out文件夹中

---

### 提取数据特征

#### 最大内接圆

```python
def circle_in(filename, img, contours_arr):
    # 计算到轮廓的距离
    raw_dist = np.empty(img.shape, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            raw_dist[i, j] = cv2.pointPolygonTest(contours_arr[1], (j, i), True)

    # 获取最大值即内接圆半径，中心点坐标
    min_val, max_val, _, max_dist_pt = cv2.minMaxLoc(raw_dist)
    max_val = abs(max_val)

    # 画出最大内接圆 避免出事
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    radius = np.int_(max_val)
    cv2.circle(result, max_dist_pt, radius, (0, 0, 255), 1, 1, 0)
    cv2.imwrite('./Out/CircleIn/' + filename, result)

    return radius * 2
```

#### 最小外接圆

```python
def circle_out(filename, img, contours_arr):
    cnt = contours_arr[1]

    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))  # 最小内接圆圆心
    radius = int(radius)  # 半径
    cv2.circle(img, center, radius, (0, 255, 0), 1)
    cv2.circle(img, center, 1, (0, 255, 0), 1)
    cv2.imwrite('./Out/CircleOut/' + filename, img)

    return radius * 2
```

#### 像素周长

```python
cv2.arcLength(contours[1], True)
```

#### 像素面积

```python
cv2.contourArea(contours[1])
```

#### 最大内接圆与最小外接圆直径比值

```python
dataDic['inscribedCircle'] / dataDic['circumscribedCircle']
```

#### 矩形度

```python
def rectangle_degree(contours_arr):
    bound_rect = cv2.minAreaRect(contours_arr[1])  # 获取最小外接矩形
    box = cv2.boxPoints(bound_rect)  # 转化为矩形点集
    area_rect = cv2.contourArea(box)
    return cv2.contourArea(contours[1]) / area_rect  # 图像面积除以矩形面积
```

#### 圆度

圆度计算公式：$ \frac{4\pi A}{P^2} $

$A$代表区域面积

$P$代表区域周长

```python
def circle_degree(contours_area, contours_length):
    return 4 * math.pi * contours_area / (contours_length ** 2)
```
