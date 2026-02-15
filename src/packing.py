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
    '''桶类：根据X坐标(time)范围管理一堆Node

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
    
class BucketManager:
    """
    桶管理器：负责将数据分桶、管理所有桶
    """
    
    def __init__(self, bucket_width: float = 2.0, dummy_radius: float = 0.5):
        """
        初始化桶管理器
        
        Args:
            bucket_width: 每个桶的宽度
            dummy_radius: 占位节点的固定半径
        """
        self.bucket_width = bucket_width
        self.dummy_radius = dummy_radius
        self.buckets: Dict[int, Bucket] = {}  # key: bucket_index
        self.category_order: List[str] = []   # 类别优先级顺序
        
        # 全局统计
        self.min_x = float('inf')
        self.max_x = float('-inf')
    
    def _get_bucket_index(self, x: float) -> int:
        """根据X坐标获取桶索引"""
        return int(math.floor(x / self.bucket_width))
    
    def _get_or_create_bucket(self, x: float) -> Bucket:
        """获取或创建桶"""
        idx = self._get_bucket_index(x)
        if idx not in self.buckets:
            x_min = idx * self.bucket_width
            x_max = x_min + self.bucket_width
            self.buckets[idx] = Bucket(x_min, x_max, self.dummy_radius)
            print(f"  创建桶 {idx}: [{x_min:.2f}, {x_max:.2f}]")
        return self.buckets[idx]
    
    def binning_data(self, circles: List[Circle]) -> None:
        """
        将数据分桶
        
        Args:
            circles: 所有原始数据圆（未放置）
        """
        print("\n" + "="*50)
        print("Step 3: 分桶逻辑 (Binning)")
        print("="*50)
        
        if not circles:
            print("  没有数据需要分桶")
            return
        
        # 计算数据的X范围（假设数据有初始x值，或者用其他特征如时间戳）
        # 这里简单使用id作为x的代理，实际应根据你的数据特征调整
        x_values = [c.id for c in circles]  # 示例：用id作为x坐标
        self.min_x = min(x_values)
        self.max_x = max(x_values)
        
        print(f"数据X范围: [{self.min_x:.2f}, {self.max_x:.2f}]")
        print(f"桶宽度: {self.bucket_width}")
        print(f"预计桶数量: {math.ceil((self.max_x - self.min_x) / self.bucket_width) + 1}")
        
        # 将每个圆放入对应的桶
        for circle in circles:
            # 假设每个圆有初始x坐标（可以是类别对应的位置）
            # 这里简单用id作为x坐标
            x_pos = circle.id * 0.5  # 示例：根据id分散
            bucket = self._get_or_create_bucket(x_pos)
            bucket.add_data_circle(circle)
        
        # 打印分桶结果
        print(f"\n分桶完成，共创建 {len(self.buckets)} 个桶:")
        for idx in sorted(self.buckets.keys()):
            bucket = self.buckets[idx]
            print(f"  桶 {idx}: {bucket}")
    
    def set_category_order(self, order: List[str]):
        """设置类别优先级顺序"""
        self.category_order = order
        print(f"\n设置类别优先级: {order}")
        
        # 对所有桶按新顺序排序
        for bucket in self.buckets.values():
            bucket.sort_by_category(order)
    
    def get_bucket_by_x(self, x: float) -> Bucket:
        """根据X坐标获取对应的桶"""
        idx = self._get_bucket_index(x)
        if idx in self.buckets:
            return self.buckets[idx]
        else:
            # 如果找不到，返回最近的桶
            if self.buckets:
                closest_idx = min(self.buckets.keys(), key=lambda i: abs(i - idx))
                print(f"  警告: 桶 {idx} 不存在，使用最近的桶 {closest_idx}")
                return self.buckets[closest_idx]
            else:
                raise ValueError("没有可用的桶")
    
    def get_next_circle_for_position(self, x: float) -> Circle:
        """
        获取指定位置的下一个要放置的圆
        
        Args:
            x: 当前位置的X坐标
        
        Returns:
            Circle: 要放置的圆
        """
        bucket = self.get_bucket_by_x(x)
        return bucket.get_next_circle(x)
    
    def print_bucket_stats(self):
        """打印所有桶的统计信息"""
        print("\n" + "="*50)
        print("桶统计信息:")
        print("="*50)
        
        total_data = 0
        total_placed = 0
        total_remaining = 0
        total_dummy = 0
        
        for idx in sorted(self.buckets.keys()):
            bucket = self.buckets[idx]
            stats = bucket.get_stats()
            print(f"桶 {idx}: {bucket}")
            print(f"  范围: [{stats['x_range'][0]:.2f}, {stats['x_range'][1]:.2f}]")
            print(f"  数据: 总计={stats['total_data']}, 已放置={stats['placed_data']}, 剩余={stats['remaining']}")
            print(f"  占位节点: {stats['dummy_count']}")
            
            total_data += stats['total_data']
            total_placed += stats['placed_data']
            total_remaining += stats['remaining']
            total_dummy += stats['dummy_count']
        
        print("-"*30)
        print(f"总计: 数据={total_data}, 已放置={total_placed}, 剩余={total_remaining}, 占位={total_dummy}")
               
class FrontChain:
    """
    前链类：封装双向链表，管理所有已放置的节点
    支持插入、删除、查找等操作
    """
    
    def __init__(self):
        """初始化空的前链"""
        self.head: Optional[Node] = None
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

'''随机圆生成器'''
def random_circle(n: int, r_min: float, r_max: float) -> List[Circle]:
    """随机生成n个半径在r_min和r_max之间的圆"""
    circles = []
    for i in range(n):
        r = random.uniform(r_min, r_max)
        circles.append(Circle(r=r, id=i))
    return circles


'''计算圆对的得分,这个算法中暂时用不上
def score(node: Node):
    a = node.circle
    b = node.next.circle
    ab = a.r + b.r
    dx = (a.x * b.r + b.x * a.r) / ab
    dy = (a.y * b.r + b.y * a.r) / ab
    return dx * dx + dy * dy
'''
'''更新链表，暂时用不上
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
'''
    

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



'''packing circle 算法，主要函数'''
def packing_circle(user_circles: List[Circle], 
                   media_circles: List[Circle], 
                   expert_circles: List[Circle]) -> List[Circle]:
    # 初始化桶
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

    


     # 打印桶信息,调试
    print("桶信息:")
    print(bucket_user)
    print(bucket_media)
    print(bucket_expert)



'''绘制函数,要修改该函数的颜色渲染条件'''
def draw_circles(circles: List[Circle], 
                 title: str = "Circle Packing Result",
                 figsize: tuple = (10, 10),
                 show_center: bool = False,
                 show_id: bool = True,
                 show_bounding: bool = False,
                 save_path: str = None,
                 wait_for_key: bool = False):
    """
    简单的圆环绘制函数，原点在画布中心
    
    Parameters:
    -----------
    circles: List[Circle] - 要绘制的圆列表
    title: str - 图表标题
    figsize: tuple - 画布大小
    show_center: bool - 是否显示圆心
    show_id: bool - 是否显示圆ID
    show_bounding: bool - 是否显示外接圆
    save_path: str - 保存路径
    wait_for_key: bool - 是否等待按键后关闭
    """
    
    # 创建画布，设置原点在中心
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    
    # 设置坐标轴，原点在中心
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    
    # 绘制坐标轴箭头
    ax.plot(1, 0, '>k', transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, '^k', transform=ax.get_xaxis_transform(), clip_on=False)
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 为圆生成颜色（使用tab10配色，简单清晰）
    colors = plt.cm.tab10(np.linspace(0, 1, len(circles)))
    
    # 绘制所有圆
    max_x = 0
    max_y = 0
    max_r = 0
    
    for i, circle in enumerate(circles):
        # 绘制圆
        patch = mpatches.Circle(
            (circle.x, circle.y), 
            circle.r,
            facecolor=colors[i],
            edgecolor='black',
            alpha=0.7,
            linewidth=1.5
        )
        ax.add_patch(patch)
        
        # 更新边界
        max_x = max(max_x, abs(circle.x) + circle.r)
        max_y = max(max_y, abs(circle.y) + circle.r)
        max_r = max(max_r, circle.r)
        
        # 显示圆心
        if show_center:
            ax.plot(circle.x, circle.y, 'k+', markersize=8)
        
        # 显示圆ID
        if show_id:
            ax.text(
                circle.x, circle.y, 
                f'{circle.id}', 
                fontsize=10, 
                ha='center', 
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
            )
    
    # 绘制外接圆
    if show_bounding and len(circles) > 0:
        # 计算外接圆心
        center_x = sum(c.x for c in circles) / len(circles)
        center_y = sum(c.y for c in circles) / len(circles)
        
        # 计算外接圆半径
        bounding_r = 0
        for circle in circles:
            distance = np.sqrt((circle.x - center_x)**2 + (circle.y - center_y)**2) + circle.r
            bounding_r = max(bounding_r, distance)
        
        # 绘制外接圆
        bounding_patch = mpatches.Circle(
            (center_x, center_y),
            bounding_r,
            facecolor='none',
            edgecolor='red',
            linestyle='--',
            linewidth=2,
            alpha=0.5
        )
        ax.add_patch(bounding_patch)
        
        # 计算并显示填充密度
        total_area = sum(np.pi * c.r * c.r for c in circles)
        bounding_area = np.pi * bounding_r * bounding_r
        density = total_area / bounding_area
        
        # 在右上角显示统计信息
        ax.text(0.95, 0.95, 
                f'Circles: {len(circles)}\nDensity: {density:.3f}',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 设置坐标轴范围
    margin = max_r * 0.5
    limit = max(max_x, max_y) + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # 等比例显示
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
     # 显示图像
    plt.show(block=False)  # 非阻塞显示
    
    if wait_for_key:
        print(f"按 N 键继续下一张图... (当前: {title})")
        while True:
            if keyboard.is_pressed('n'):
                plt.close()
                print("加载下一张图...")
                break
            plt.pause(0.1)  # 避免CPU过载
    else:
        plt.show()  # 阻塞显示

def main():
    circles = []
    circles_list = []
    # 随机生成圆三组
    for i in range(1, 4):
        circles_tmp = random_circle(10, 0.1, 0.5)
        for j in range(len(circles_tmp)):
            circles_list.append(Circle(r=circles_tmp[j].r, id=i*10+j, category=Category(i), tim=random.uniform(0, 3)))
        circles.append(circles_list)
        circles_list = []  # 重置列表

    # print(f"生成的圆:{circles}")

    packing_circle(circles[0], circles[1], circles[2])


if __name__ == "__main__":
    
    main()
