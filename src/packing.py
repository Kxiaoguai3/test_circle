'''
作者：夏月华
日期：2026-2-15
描述：圆packing算法实现，包含以下主要类和函数：
- Circle: 圆类，存储圆心坐标、半径、类别、
- Node: 双向链表节点类
- Tim: 时间类，将圆按照时间划分，装入类的data中
- Bucket: 桶类：根据X坐标(time)范围分类管理即将计算的圆
- BucketManager: 桶管理器：负责将数据分桶、管理所有桶（可以不要了）

备注：packing_circle函数完成分桶，下一步开始实现第一批前链的创建和堆叠逻辑。
     绘制函数需要修改颜色渲染条件，以区分不同类别的圆。
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import random
import math
import time
import keyboard
from enum import Enum

class Category(Enum):

    USER = 1
    MEDIA = 2
    EXPERT = 3
    UNKOWN = 0

class Circle:
    """圆类，存储圆心坐标和半径"""
    def __init__(self, x: float = 0.0, y: float = 0.0, r: float = 0.0, id: int = 0
                 , category: Category = Category.UNKOWN, tim: float = 0.0, is_placeholder: bool = False):
        '''
        __init__ 的 Docstring
        
        :param category: 类别
        :param tim: 该信息出现的时间
        :param is_placeholder: 是否为占位圆
        '''
        self.x = x
        self.y = y
        self.r = r
        self.id = id
        self.category = category
        self.tim = tim
        self.is_placeholder = is_placeholder

    def __repr__(self):
        return f"\n圆 {self.id}: ({self.x:.2f}, {self.y:.2f}), r={self.r:.2f}, 类别={self.category}, 时间={self.tim:.2f}"

class Node:
    """双向链表节点类"""
    def __init__(self, circle: Circle):
        self.circle = circle
        self.prev = None
        self.next = None

class Tim:
    '''时间类，将圆按照时间划分，装入类的data中'''
    def __init__(self, name: str, data: List = [Circle], border: float = 0.0):
        self.name = name
        self.data = data
        self.border = border

    def __repr__(self):
        return f"\n{self.name}: 数量={len(self.data)}，数据={self.data}"

class Bucket:
    '''桶类：根据X坐标(time)范围分类管理即将计算的圆

        Bucket{
            t1{
                c1,c2...
            },

            t2{
            c1,c2...
            },

            t3{
                c1,c2...
            }
            ...
        }
    '''
    def __init__(self, category: Category = Category.UNKOWN, tim: List = [Tim]):
        # 桶的信息
        self.category = category

        # 该桶的X坐标范围（根据时间划分）  
        self.tim = tim
        
        # 统计信息
        self.total_data_count = 0         # 总数据量
        self.placed_data_count = 0        # 已放置数据量
        self.dummy_count = 0              # 空闲节点数量

    def add_data_circle(self, circle: Circle):
        '''
            将数据圆添加到桶中
        :param circle: 放入的圆
        '''
        if not circle.category == self.category:
            print(f"错误: 将类别 {circle.category} 的圆添加到类别 {self.category} 的桶中")
            return
        
        else:
            for t in self.tim:
                if t.border >= circle.tim:
                    t.data.append(circle)
                    self.total_data_count += 1
                    break
                # 最后一个tim
                elif t == self.tim[-1]:
                    t.data.append(circle)
                    self.total_data_count += 1

    def sort_by_circle(self):
        """
        将桶中每一个圆按照r从大到小排序
        """
        for t in self.tim:
            t.data.sort(key=lambda c: c.r, reverse=True)
    
    def has_data(self) -> bool:
        """检查是否还有数据节点"""
        return len(self.data_queue) > 0
    
    def get_remaining_count(self) -> int:
        """获取剩余数据节点数量"""
        return len(self.data_queue)
    
    def __repr__(self):
        return f"桶 {self.category} [0, {self.tim[-1].border:.2f}], 总数据量={self.total_data_count},数据队列={self.tim}\n\n"
                  
class FrontChain:
    """
    前链类：封装双向链表，管理所有已放置的节点
    支持插入、删除、查找等操作
    """
    
    def __init__(self):
        """初始化空的前链"""
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self.size: int = 0
        self.active_nodes: Dict[int, Node] = {}  # id到节点的映射，用于快速查找
        self.deleted_nodes: List[int] = []  # 已删除节点的ID列表
        
        print("创建新的前链")
    
    def is_empty(self) -> bool:
        """检查前链是否为空"""
        return self.head is None
    
    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """根据ID获取节点"""
        return self.active_nodes.get(node_id)
    
    def remove(self, node: Node) -> bool:
        """
        从前链中移除节点（真正的删除）
        
        Args:
            node: 要移除的节点
        
        Returns:
            bool: 是否成功移除
        """
        if self.is_empty() or node is None:
            print("  错误: 前链为空或节点为空")
            return False
        
        if not node.is_active:
            print(f"  节点 {node.circle.id} 已不在前链中")
            return False
        
        node_id = node.circle.id
        print(f"  移除节点 {node_id}")
        
        if self.size == 1:
            # 只有一个节点
            if node == self.head:
                self.head = None
            else:
                print(f"  错误: 节点不是head但链表只有一个节点")
                return False
        else:
            # 调整前后节点的指针
            prev_node = node.prev
            next_node = node.next
            
            prev_node.next = next_node
            next_node.prev = prev_node
            
            # 如果移除的是head，更新head
            if node == self.head:
                self.head = next_node
        
        # 真正的删除：清理节点的指针，标记为非活跃
        node.prev = None
        node.next = None
        node.is_active = False
        
        # 从活跃节点映射中移除
        if node_id in self.active_nodes:
            del self.active_nodes[node_id]
        
        # 记录已删除的ID
        self.deleted_nodes.append(node_id)
        self.size -= 1
        
        print(f"  删除完成，前链当前大小: {self.size}")
        return True
    
    def insert_after(self, ref_node: Node, new_node: Node) -> Node:
        """
        在参考节点后插入新节点
        
        Args:
            ref_node: 参考节点
            new_node: 要插入的新节点
        
        Returns:
            Node: 插入的节点
        """
        print(f"  在节点 {ref_node.circle.id} 后插入新节点 {new_node.circle.id}")
        
        if self.is_empty():
            # 空链表，创建链表
            self.head = new_node
            new_node.prev = None
            new_node.next = None
        else:
            # 在ref_node后插入
            next_node = ref_node.next
            
            new_node.prev = ref_node
            new_node.next = next_node
            ref_node.next = new_node
            self.remove(next_node)
            next_node.prev = new_node
            
            # 如果ref_node是head，不需要更新head
        
        # 标记为活跃
        new_node.is_active = True
        self.active_nodes[new_node.circle.id] = new_node
        self.size += 1
        
        return new_node
    
    def insert_before(self, ref_node: Node, new_node: Node) -> Node:
        """
        在参考节点前插入新节点
        
        Args:
            ref_node: 参考节点
            new_node: 要插入的新节点
        
        Returns:
            Node: 插入的节点
        """
        print(f"  在节点 {ref_node.circle.id} 前插入新节点 {new_node.circle.id}")
        
        if self.is_empty():
            return self.insert_after(ref_node, new_node)
        else:
            prev_node = ref_node.prev
            return self.insert_after(prev_node, new_node)
    
    def find_lowest_y(self) -> Optional[Node]:
        """
        找到前链中y值最小的节点
        
        Returns:
            Optional[Node]: y值最小的节点
        """
        if self.is_empty():
            return None
        
        current = self.head
        lowest = self.head
        min_y = self.head.circle.y
        
        while True:
            if current.circle.y < min_y:
                min_y = current.circle.y
                lowest = current
            
            current = current.next
            if current.next == None:
                break
        
        print(f"  找到最低点: 节点 {lowest.circle.id}, y={min_y:.3f}")
        return lowest
    
    def find_nearest_by_x(self, ref_node: Node) -> Tuple[Node, Node]:
        """
        找到参考节点X方向最近的两个节点（prev和next中X距离较小的）
        
        Args:
            ref_node: 参考节点
        
        Returns:
            Tuple[Node, Node]: (c1, c2)，c1和c2是相邻节点对
        """
        prev_dist = abs(ref_node.prev.circle.x - ref_node.circle.x)
        next_dist = abs(ref_node.next.circle.x - ref_node.circle.x)
        
        if prev_dist < next_dist:
            print(f"  X距离: prev={prev_dist:.3f} < next={next_dist:.3f}, 选择prev")
            return ref_node.prev, ref_node
        else:
            print(f"  X距离: next={next_dist:.3f} <= prev={prev_dist:.3f}, 选择next")
            return ref_node, ref_node.next
    
    def traverse(self) -> List[Node]:
        """
        遍历前链，返回所有节点列表
        
        Returns:
            List[Node]: 节点列表（按顺序）
        """
        if self.is_empty():
            return []
        
        nodes = []
        current = self.head
        
        while True:
            nodes.append(current)
            current = current.next
            if current == self.head:
                break
        
        return nodes
    
    def print_chain(self, title: str = "前链状态"):
        """打印前链的所有节点"""
        print(f"\n{title}:")
        if self.is_empty():
            print("  前链为空")
            return
        
        nodes = self.traverse()
        print(f"  大小: {self.size}, 节点顺序:")
        for i, node in enumerate(nodes):
            active_status = "活跃" if node.is_active else "已删除"
            print(f"    {i+1}. id={node.circle.id}, 类别={node.circle.category}, "
                  f"位置=({node.circle.x:.2f}, {node.circle.y:.2f}), 状态={active_status}")
        
        # 打印已删除节点
        if self.deleted_nodes:
            print(f"  已删除节点: {self.deleted_nodes}")
    
    def validate_chain(self) -> bool:
        """
        验证链表的完整性
        
        Returns:
            bool: 链表是否有效
        """
        if self.is_empty():
            return True
        
        # 检查循环性
        current = self.head
        count = 0
        
        while count < self.size:
            if current.next.prev != current:
                print(f"  错误: 节点 {current.circle.id} 的next.prev不指向自己")
                return False
            if current.prev.next != current:
                print(f"  错误: 节点 {current.circle.id} 的prev.next不指向自己")
                return False
            
            current = current.next
            count += 1
        
        if current != self.head:
            print(f"  错误: 遍历后没有回到head")
            return False
        
        if count != self.size:
            print(f"  错误: 遍历计数 {count} != 链表大小 {self.size}")
            return False
        
        print("  链表验证通过 ✅")
        return True
    
    '''一个圆是否在前链中与一个圆相交'''
    def circle_intersect(self, circle: Circle) -> Tuple[bool, Optional[Node]]:
        '''
        判断一个圆是否与前链中的任何一个圆相交

        Args:
            circle: 要检测的圆

        Returns:
            bool: 相交返回True，否则返回False
            Optional[Node]: 如果相交，返回相交的节点；否则返回None
        '''

        node = self.head
        while node is not None:
            if circles_intersect(node.circle, circle):
                return True, node
            node = node.next
        return False, None

    def update_head(self, new_head: Node, next_node: Node):
        """更新head指针"""
        self.head = new_head
        self.head.next = next_node
        next_node.prev = self.head
        self.size += 1
        self.head.is_active = True
        self.active_nodes[self.head.circle.id] = self.head
    
    def update_tail(self, new_tail: Node, prev_node: Node):
        """更新tail指针"""
        self.tail = new_tail
        self.size += 1
        self.tail.is_active = True
        self.active_nodes[self.tail.circle.id] = self.tail
        prev_node.next = self.tail
        self.tail.prev = prev_node

'''随机圆生成器'''
def random_circle(n: int, r_min: float, r_max: float) -> List[Circle]:
    """随机生成n个半径在r_min和r_max之间的圆"""
    circles = []
    for i in range(n):
        r = random.uniform(r_min, r_max)
        circles.append(Circle(r=r, id=i))
    return circles
    
'''通过两个圆，计算第三个相切圆,计算出来的圆出现在向量ab的左侧'''
def place_circle(b: Circle, a: Circle, c: Circle):
    print(f"计算 id:{c.id} 的位置，已知 id:{a.id} 和 id:{b.id} 的位置")
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

'''通过一个圆和垂直边界，计算第三个圆'''
def place_circle_vertical(circle: Circle, ref_circle: Circle, 
                          border_x: float, side: str = "left") -> None:
    """
    放置一个圆，使其与参考圆相切，并且与垂直边界相切
    
    Args:
        circle: 要放置的圆
        ref_circle: 参考圆
        border_x: 垂直边界的X坐标
        side: 圆在边界的哪一侧，"right" 或 "left"
    
    说明：
        圆会与参考圆相切，同时与垂直边界相切
        圆的Y坐标与参考圆相同
    """
    # 计算两圆半径之和
    total_r = ref_circle.r + circle.r
    
    if side == "right":
        # 圆在边界右侧，与右边界相切
        circle.x = border_x - circle.r
        # 确保与参考圆相切
        dx = abs(circle.x - ref_circle.x)
        if dx > total_r:
            # 如果距离太远，调整位置
            circle.x = ref_circle.x + total_r if ref_circle.x < circle.x else ref_circle.x - total_r
    else:  # side == "left"
        # 圆在边界左侧，与左边界相切
        circle.x = border_x + circle.r
        dx = abs(circle.x - ref_circle.x)
        if dx > total_r:
            circle.x = ref_circle.x - total_r if ref_circle.x > circle.x else ref_circle.x + total_r
    
    # Y坐标与参考圆相同（相切于同一点）
    circle.y = ref_circle.y

'''通过一个圆和水平边界，计算第三个圆'''
def place_circle_horizontal(circle: Circle, ref_circle: Circle, 
                            border_y: float, side: str = "top") -> None:
    """
    放置一个圆，使其与参考圆相切，并且与水平边界相切
    
    Args:
        circle: 要放置的圆
        ref_circle: 参考圆
        border_y: 水平边界的Y坐标
        side: 圆在边界的哪一侧，"top" 或 "bottom"
    
    说明：
        圆会与参考圆相切，同时与水平边界相切
        圆的X坐标与参考圆相同
    """
    # 计算两圆半径之和
    total_r = ref_circle.r + circle.r
    
    if side == "top":
        # 圆在边界上方，与上边界相切
        circle.y = border_y - circle.r
        dy = abs(circle.y - ref_circle.y)
        if dy > total_r:
            circle.y = ref_circle.y + total_r if ref_circle.y < circle.y else ref_circle.y - total_r
    else:  # side == "bottom"
        # 圆在边界下方，与下边界相切
        circle.y = border_y + circle.r
        dy = abs(circle.y - ref_circle.y)
        if dy > total_r:
            circle.y = ref_circle.y - total_r if ref_circle.y > circle.y else ref_circle.y + total_r
    
    # X坐标与参考圆相同
    circle.x = ref_circle.x

'''判断圆是否与垂直线相交'''
def circle_intersect_vertical(circle: Circle, x_line: float, epsilon: float = 1e-6) -> bool:
    """
    判断圆是否与垂直线相交
    
    Args:
        circle: 要检测的圆
        x_line: 垂直线的X坐标
        epsilon: 误差容限
    
    Returns:
        bool: 相交返回True，否则返回False
    
    说明：
        圆与垂直线相交的条件：圆心到直线的距离 <= 半径
        距离 = |circle.x - x_line|
    """
    # 圆心到垂直线的距离
    distance = abs(circle.x - x_line)
    
    # 相交条件：距离 <= 半径（考虑微小误差）
    return distance <= circle.r + epsilon

'''判断圆是否与水平线相交'''
def circle_intersect_horizontal(circle: Circle, y_line: float, epsilon: float = 1e-6) -> bool:
    """
    判断圆是否与水平线相交
    
    Args:
        circle: 要检测的圆
        y_line: 水平线的Y坐标
        epsilon: 误差容限
    
    Returns:
        bool: 相交返回True，否则返回False
    
    说明：
        圆与水平线相交的条件：圆心到直线的距离 <= 半径
        距离 = |circle.y - y_line|
    """
    # 圆心到水平线的距离
    distance = abs(circle.y - y_line)
    
    # 相交条件：距离 <= 半径（考虑微小误差）
    return distance <= circle.r + epsilon

'''packing circle 算法，主要函数'''
def packing_circle(user_circles: List[Circle], 
                   media_circles: List[Circle], 
                   expert_circles: List[Circle]) -> List[Bucket]:
    # 初始化桶，暂时用3个时段来区分
    tim_num = 3
    # user
    bucket_user = Bucket(category=Category.USER, tim = [Tim(name="t1", data=[], border=1)])
    bucket_user.tim.append(Tim(name="t2", data=[], border=2))
    bucket_user.tim.append(Tim(name="t3", data=[], border=3))

    # media
    bucket_media = Bucket(category=Category.MEDIA, tim = [Tim(name="t1", data=[], border=1)])
    bucket_media.tim.append(Tim(name="t2", data=[], border=2))
    bucket_media.tim.append(Tim(name="t3", data=[], border=3))

    # expert
    bucket_expert = Bucket(category=Category.EXPERT, tim = [Tim(name="t1", data=[], border=1)])
    bucket_expert.tim.append(Tim(name="t2", data=[], border=2))
    bucket_expert.tim.append(Tim(name="t3", data=[], border=3))

    # 将圆放入对应的桶中
    for circle in user_circles:
        bucket_user.add_data_circle(circle)
    for circle in media_circles:
        bucket_media.add_data_circle(circle)
    for circle in expert_circles:
        bucket_expert.add_data_circle(circle)

   

    # 桶内排序
    bucket_user.sort_by_circle()
    bucket_media.sort_by_circle()
    bucket_expert.sort_by_circle()

    # 将三个桶放入到列表中，依次放置
    bucket_list = [bucket_user, bucket_media, bucket_expert]

    # 创建空前链
    front_chain = FrontChain()

    # 放置不同之间段
    for i in range(tim_num):

        # 分类放置
        for bucket in bucket_list:
            is_bottom_ready = False

            # 放置圆
            for circle in bucket.tim[i].data:
                # 判断是否是空链
                if front_chain.is_empty():
                    # 放到相应坐标的右上角,并放入前链
                    place_circle_border(circle, (bucket.tim[i].border - bucket.tim[0].border, 0))
                    front_chain.head = Node(circle)
                    front_chain.tail = Node(circle)

                # 最下一层没有铺好
                elif not is_bottom_ready:
                    # 计算前链上最后一个元圆与水平线都相切的圆
                    place_circle_horizontal(circle, front_chain.tail.circle, bucket.tim[i].border, side="top")
                    # 检测是否与边界相交
                    if not circle_intersect_vertical(circle, bucket.tim[i].border):
                        # 放入到前链尾
                        tmp_node = Node(circle)
                        front_chain.tail.next = tmp_node
                        tmp_node.prev = front_chain.tail
                        front_chain.tail = tmp_node
                    else:
                        is_bottom_ready = True  # 标记为铺好了
                        # 重新计算该圆的位置，放到边界上
                        place_circle_vertical(circle, front_chain.tail.circle, bucket.tim[i].border)
                        # 检查是否与前链相交
                        is_intersect, intersect_node = front_chain.circle_intersect(circle)
                        if is_intersect:
                            # 如果相交，重新计算该圆的位置，放到垂直边界上,成功为止
                            while is_intersect:
                                place_circle_vertical(circle, intersect_node.circle, bucket.tim[i].border)
                                is_intersect, intersect_node = front_chain.circle_intersect(circle)
                        # 放入到前链尾
                        tmp_node = Node(circle)
                        front_chain.tail.next = tmp_node
                        tmp_node.prev = front_chain.tail
                        front_chain.tail = tmp_node

                # 最下一层已经铺好
                else:
                    # 找到前链上最低的节点
                    miny_node = front_chain.find_lowest_y()
                    # 找到离该节点最近的节点
                    a, b = front_chain.find_nearest_by_x(miny_node)
                    # 计算第三个相切圆的位置
                    place_circle(b.circle, a.circle, circle)
                    # 检测是否与边界相交,左边，右边
                    # 右边
                    if circle_intersect_vertical(circle, bucket.tim[i].border):
                        # 让右边的圆和有边界计算相交圆
                        place_circle_vertical(circle, b.circle, bucket.tim[i].border)
                        # 检查是否与前链相交
                        is_intersect, intersect_node = front_chain.circle_intersect(circle)
                        if is_intersect:
                            # 如果相交，重新计算该圆的位置，放到垂直边界上,成功为止
                            while is_intersect:
                                place_circle_vertical(circle, intersect_node.circle, bucket.tim[i].border)
                                is_intersect, intersect_node = front_chain.circle_intersect(circle)

                        # 放入到前链尾
                        front_chain.update_tail(Node(circle), b)
                    # 左边
                    elif circle_intersect_vertical(circle, bucket.tim[i].border - bucket.tim[0].border):
                        # 让左边的圆和左边界计算相交圆
                        place_circle_vertical(circle, a.circle, bucket.tim[i].border - bucket.tim[0].border)
                        # 检查是否与前链相交
                        is_intersect, intersect_node = front_chain.circle_intersect(circle)
                        if is_intersect:
                            # 如果相交，重新计算该圆的位置，放到垂直边界上,成功为止
                            while is_intersect:
                                place_circle_vertical(circle, intersect_node.circle, bucket.tim[i].border - bucket.tim[0].border)
                                is_intersect, intersect_node = front_chain.circle_intersect(circle)
                        # 放入到前链头
                        front_chain.update_head(Node(circle), a)
                    # 检查前链
                    else:
                        # 检查是否与前链相交
                        is_intersect, intersect_node = front_chain.circle_intersect(circle)
                        if is_intersect:
                            # 如果相交，重新计算该圆的位置，放到垂直边界上,成功为止
                            while is_intersect:
                                place_circle_vertical(circle, intersect_node.circle, bucket.tim[i].border - bucket.tim[0].border)
                                is_intersect, intersect_node = front_chain.circle_intersect(circle)
                        
                        # 放入到前链
                        front_chain.insert_after(a, Node(circle))


#  # 打印桶信息,调试
    # print("桶信息:")
    # print(bucket_user)
    # print(bucket_media)
    # print(bucket_expert)

                    
    return bucket_list

def draw_circles(buckets: List[Bucket], 
                 title: str = "Circle Packing Result",
                 figsize: tuple = (20, 20),
                 show_id: bool = False,
                 show_tim_borders: bool = True,
                 save_path: str = None,
                 wait_for_key: bool = False):
    """
    绘制圆 packing 结果，按类别显示不同颜色
    
    Parameters:
    -----------
    buckets: List[Bucket] - 桶列表，每个桶包含多个时间段的圆
    title: str - 图表标题
    figsize: tuple - 画布大小
    show_id: bool - 是否显示圆ID
    show_tim_borders: bool - 是否显示时间段边界线
    save_path: str - 保存路径
    wait_for_key: bool - 是否等待按键后关闭
    """
    
    # 创建画布，原点设置在左下角
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    
    # 设置坐标轴，原点在左下角
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # 设置坐标轴范围
    ax.set_xlim(0, None)  # X轴从0开始
    ax.set_ylim(0, None)  # Y轴从0开始
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 定义类别颜色映射
    category_colors = {
        Category.USER: '#FF6B6B',     # 珊瑚红
        Category.MEDIA: '#4ECDC4',     # 薄荷绿
        Category.EXPERT: '#45B7D1',    # 天空蓝
        Category.UNKOWN: '#95A5A6'     # 灰色
    }
    
    # 收集所有圆用于计算边界
    all_circles = []
    for bucket in buckets:
        for tim in bucket.tim:
            all_circles.extend(tim.data)
    
    if not all_circles:
        print("no circles to draw")
        return
    
    # 绘制所有圆
    max_x = 0
    max_y = 0
    
    for bucket in buckets:
        for tim in bucket.tim:
            # 获取该时间段的边界线位置
            border_x = tim.border
            
            # 绘制时间段边界线（虚线）
            if show_tim_borders:
                ax.axvline(x=border_x, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                # 在边界线上方标注时间段名称
                ax.text(border_x, max_y * 0.95, tim.name, 
                       fontsize=9, ha='center', va='top',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # 绘制该时间段的圆
            for circle in tim.data:
                # 获取该类别对应的颜色
                color = category_colors.get(circle.category, category_colors[Category.UNKOWN])
                
                # 绘制圆
                patch = mpatches.Circle(
                    (circle.x, circle.y), 
                    circle.r,
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7,
                    linewidth=1.2
                )
                ax.add_patch(patch)
                
                # 更新最大坐标
                max_x = max(max_x, circle.x + circle.r)
                max_y = max(max_y, circle.y + circle.r)
                
                # 显示圆ID
                if show_id:
                    ax.text(
                        circle.x, circle.y, 
                        f'{circle.id}', 
                        fontsize=8, 
                        ha='center', 
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8)
                    )
    
    # 设置坐标轴范围，留出边距
    margin = max(max_x, max_y) * 0.1
    ax.set_xlim(-margin, max_x + margin)
    ax.set_ylim(-margin, max_y + margin)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('time →', fontsize=12)
    ax.set_ylabel('height →', fontsize=12)
    
    # 添加图例
    legend_elements = []
    for category, color in category_colors.items():
        if category != Category.UNKOWN:  # 不显示未知类别在图例中
            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor='black', 
                             label=f'{category.name}', alpha=0.7)
            )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 添加统计信息
    total_circles = len(all_circles)
    category_counts = {}
    for circle in all_circles:
        category_counts[circle.category] = category_counts.get(circle.category, 0) + 1
    
    stats_text = f'总圆数: {total_circles}\n'
    for category, count in category_counts.items():
        if category != Category.UNKOWN:
            stats_text += f'{category.name}: {count}\n'
    
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 等比例显示
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    # 显示图像
    plt.show(block=False)
    
    if wait_for_key:
        print(f"按 N 键继续下一张图... (当前: {title})")
        while True:
            if keyboard.is_pressed('n'):
                plt.close()
                print("加载下一张图...")
                break
            plt.pause(0.1)
    else:
        plt.show()

'''计算圆与边界相切，圆的数据'''
def place_circle_border(circle: Circle, border: Tuple[float, float]):
    """根据一个圆的r和两个边界位置，算出该圆圆心的x，y
        将圆放置在两条线的右上角"""
    circle.x = border[0] + circle.r
    circle.y = border[1] + circle.r


def main():
    # 随机生成圆
    circles = []
    circles_list = []
    # 随机生成圆三组
    for i in range(1, 4):
        circles_tmp = random_circle(30, 0.01, 0.05)
        for j in range(len(circles_tmp)):
            circles_list.append(Circle(r=circles_tmp[j].r, id=i*10+j, category=Category(i), tim=random.uniform(0, 3)))
        circles.append(circles_list)
        circles_list = []  # 重置列表

    # print(f"生成的圆:{circles}")

    # 调用packing算法
    bucket_list = packing_circle(circles[0], circles[1], circles[2])

    # 绘制
    draw_circles(bucket_list)



if __name__ == "__main__":
    
    main()
