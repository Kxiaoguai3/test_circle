'''
作者：夏月华


by_myself 的 Docstring

通过自己理解，写一个packing circle的程序。

算法：
1、使用循环双向链表实现一个环形链表。
2、初始化先放三个相切的圆，并放入链表中
3、遍历前链表，计算得分（通过计算两个圆的加权质心到原点的距离平方）最低的圆对。
4、通过计算出来的圆对a、b算出第三个圆的坐标，然后更新链表。
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import random
import math
import time

class Circle:
    """圆类，存储圆心坐标和半径"""
    def __init__(self, x: float = 0.0, y: float = 0.0, r: float = 0.0, id: int = 0):
        self.x = x
        self.y = y
        self.r = r
        self.id = id

class Node:
    """双向链表节点类"""
    def __init__(self, circle: Circle):
        self.circle = circle
        self.prev = None
        self.next = None

'''随机圆生成器'''
def random_circle(n: int, r_min: float, r_max: float) -> List[Circle]:
    """随机生成n个半径在r_min和r_max之间的圆"""
    circles = []
    for i in range(n):
        r = random.uniform(r_min, r_max)
        circles.append(Circle(r=r, id=i))
    return circles

'''计算圆对的得分'''
def score(node: Node):
    a = node.circle
    b = node.next.circle
    ab = a.r + b.r
    dx = (a.x * b.r + b.x * a.r) / ab
    dy = (a.y * b.r + b.y * a.r) / ab
    return dx * dx + dy * dy

'''更新链表'''
def update_list(node: Node, circle: Circle) -> Node:
    """更新链表，将circle插入到node之后"""
    a = node
    b = node.next
    p = Node(circle)

    a.next = p
    p.prev = a
    p.next = b
    b.prev = p

    return p

'''通过两个圆，计算第三个相切圆'''
def place_circle(b: Circle, a: Circle, c: Circle):
    """根据两个已知圆的位置，计算第三个相切圆的位置"""
    dx = b.x - a.x
    dy = b.y - a.y
    d2 = dx * dx + dy * dy
    
    if d2 > 0:
        a2 = a.r + c.r
        a2 *= a2
        b2 = b.r + c.r
        b2 *= b2
        
        if a2 > b2:
            x = (d2 + b2 - a2) / (2 * d2)
            y = math.sqrt(max(0, b2 / d2 - x * x))
            c.x = b.x - x * dx - y * dy
            c.y = b.y - x * dy + y * dx
        else:
            x = (d2 + a2 - b2) / (2 * d2)
            y = math.sqrt(max(0, a2 / d2 - x * x))
            c.x = a.x + x * dx - y * dy
            c.y = a.y + x * dy + y * dx
    else:
        c.x = a.x + c.r
        c.y = a.y

'''两个圆是否相交'''
def circles_intersect(a: Circle, b: Circle) -> bool:
    """判断两个圆是否相交（考虑微小误差）"""
    dr = a.r + b.r - 1e-6
    dx = b.x - a.x
    dy = b.y - a.y
    # 相交返回True，否则返回False
    return dr > 0 and dr * dr > dx * dx + dy * dy

'''遍历链表，查看是否有相交'''
def check_list(a_node:Node, b_node:Node, circle:Circle) -> Tuple[Node, Node, bool]:
    tmp_a = a_node.prev
    tmp_b = b_node.next
    while True:
        if not circles_intersect(tmp_a.circle, circle):
            tmp_a = a_node.prev
        else:
            return tmp_a, b_node, True
            
        if not circles_intersect(tmp_b.circle, circle):
            tmp_b = b_node.next
        else:
            return a_node, tmp_b, True
            
        if tmp_a == a_node or tmp_b == b_node:
            return a_node, b_node, False
        
'''计算前链中得分最低的圆对'''
def best_sore(start_node: Node) -> Tuple[Node, Node]:
    """遍历链表，找到得分最低的相邻圆对"""
    current = start_node
    best_node = current
    best_score = score(current)
    
    # 遍历整个链表
    current = current.next
    while current != start_node:
        current_score = score(current)
        if current_score < best_score:
            best_score = current_score
            best_node = current
        current = current.next
    
    # best_node 和 best_node.next 就是最佳圆对
    return best_node, best_node.next

'''packing circle 算法，主要函数'''
def packing_circle(circles: List[Circle]) -> float:
    """packing circle算法，打包n个圆"""
    n = len(circles)
    if n == 0:
        return 0.0
    
    # 先放三个相切的圆

    # 1. 放置第一个圆
    a = circles[0]
    a.x, a.y = 0, 0
    if n == 1:
        return a.r
    
    # 2. 放置第二个圆
    b = circles[1]
    a.x = -b.r
    b.x = a.r
    b.y = 0
    if n == 2:
        return a.r + b.r
    
    # 3. 放置第三个圆
    c = circles[2]
    place_circle(b, a, c)

    # 创建环形链表
    
    a_node = Node(a)
    b_node = Node(b)
    c_node = Node(c)

    a_node.next = b_node
    a_node.prev = c_node
    b_node.prev = a_node
    b_node.next = c_node
    c_node.prev = b_node
    c_node.next = a_node

    # 遍历链表，计算得分最低的圆对
    best_a, best_b = best_sore(a_node)

    for i in range(3, n):
        new_circle = Circle(r=circles[i].r, id=circles[i].id)
        inserted = False

        # 从此处开始尝试
        current_a = best_a
        current_b = best_b

        place_circle(current_b.circle, current_a.circle, new_circle)

        # 遍历链表，查看新圆是否有相交
        current_a, current_b, inserted = check_list(current_a, current_b, new_circle)
        if inserted:
            update_list(current_a, new_circle)
        else:
            # 尝试新的圆对，只到成功为止,并且记录是否超时，超时则警告，然后退出
            start_time = time.time()
            while inserted:
               place_circle(current_b.circle, current_a.circle, new_circle)
               current_a, current_b, inserted = check_list(current_a, current_b, new_circle)
               now_time = time.time()
               if now_time - start_time > 10:
                print(f"id:{new_circle.id} 超时！")
                break
        print(f"id:{new_circle.id}完成")



            

            
        

        
    










