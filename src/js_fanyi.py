# js文件翻译版
import math
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Circle:
    """圆的数据结构"""
    x: float = 0.0
    y: float = 0.0
    r: float = 0.0

class Node:
    """链表节点"""
    def __init__(self, circle: Circle):
        self.circle = circle
        self.next: Optional['Node'] = None
        self.previous: Optional['Node'] = None

class RandomGenerator:
    """线性同余随机数生成器"""
    def __init__(self, seed: int = 1):
        self.seed = seed
        self.modulus = 4294967296
        self.multiplier = 1664525
        self.increment = 1013904223
    
    def next(self) -> float:
        """生成下一个随机数 [0, 1)"""
        self.seed = (self.multiplier * self.seed + self.increment) % self.modulus
        return self.seed / self.modulus

def place_circle(a: Circle, b: Circle, c: Circle) -> None:
    """
    基于两个已知圆放置第三个圆
    """
    dx = b.x - a.x
    dy = b.y - a.y
    d2 = dx * dx + dy * dy
    
    if d2 > 1e-12:
        a2 = a.r + c.r
        a2 *= a2
        b2 = b.r + c.r
        b2 *= b2
        
        if a2 > b2:
            x = (d2 + b2 - a2) / (2.0 * d2)
            y = math.sqrt(max(0.0, b2 / d2 - x * x))
            c.x = b.x - x * dx - y * dy
            c.y = b.y - x * dy + y * dx
        else:
            x = (d2 + a2 - b2) / (2.0 * d2)
            y = math.sqrt(max(0.0, a2 / d2 - x * x))
            c.x = a.x + x * dx - y * dy
            c.y = a.y + x * dy + y * dx
    else:
        print("进入到奇怪的地方")
        c.x = a.x + c.r
        c.y = a.y

def circles_intersect(a: Circle, b: Circle, epsilon: float = 1e-6) -> bool:
    """检查两个圆是否相交"""
    dr = a.r + b.r - epsilon
    dx = b.x - a.x
    dy = b.y - a.y
    return dr > 0 and dr * dr > dx * dx + dy * dy

def node_score(node: Node) -> float:
    """计算节点得分"""
    a = node.circle
    b = node.next.circle
    ab = a.r + b.r
    dx = (a.x * b.r + b.x * a.r) / ab
    dy = (a.y * b.r + b.y * a.r) / ab
    return dx * dx + dy * dy

def find_enclosing_circle(circles: List[Circle], random_gen: RandomGenerator) -> Circle:
    """寻找包围圆（简化实现）"""
    if not circles:
        return Circle(0, 0, 0)
    
    # 计算加权中心
    total_x = 0.0
    total_y = 0.0
    total_weight = 0.0
    max_distance = 0.0
    
    for circle in circles:
        weight = circle.r
        total_x += circle.x * weight
        total_y += circle.y * weight
        total_weight += weight
    
    center_x = total_x / total_weight
    center_y = total_y / total_weight
    
    # 找到最远的点
    for circle in circles:
        dx = circle.x - center_x
        dy = circle.y - center_y
        distance = math.sqrt(dx * dx + dy * dy) + circle.r
        max_distance = max(max_distance, distance)
    
    return Circle(center_x, center_y, max_distance)

def pack_siblings_random(circles: List[Circle], random_gen: RandomGenerator) -> float:
    """
    随机圆打包算法
    
    参数:
        circles: 圆列表，每个圆需要有半径r属性
        random_gen: 随机数生成器
        
    返回:
        包围圆的半径
    """
    n = len(circles)
    if n == 0:
        return 0.0
    
    # 放置第一个圆
    circles[0].x = 0.0
    circles[0].y = 0.0
    if n == 1:
        return circles[0].r
    
    # 放置第二个圆
    circles[1].x = circles[0].r + circles[1].r
    circles[1].y = 0.0
    circles[0].x = -circles[1].r
    if n == 2:
        return circles[0].r + circles[1].r
    
    # 放置第三个圆
    place_circle(circles[1], circles[0], circles[2])
    
    # 创建初始链表
    a = Node(circles[0])
    b = Node(circles[1])
    c = Node(circles[2])
    
    a.next = c
    a.previous = b
    b.next = a
    b.previous = c
    c.next = b
    c.previous = a
    
    # 放置剩余的圆
    i = 3
    while i < n:
        new_circle = circles[i]
        
        # 尝试放置新圆
        place_circle(a.circle, b.circle, new_circle)
        new_node = Node(new_circle)
        
        # 在链上寻找插入位置
        j = b.next
        k = a.previous
        sj = b.circle.r
        sk = a.circle.r
        placed = False
        
        while True:
            if sj <= sk:
                if circles_intersect(j.circle, new_circle):
                    # 相交，回退并重新开始
                    b = j
                    a.next = b
                    b.previous = a
                    i -= 1  # 重新尝试这个圆
                    placed = True
                    break
                sj += j.circle.r
                j = j.next
            else:
                if circles_intersect(k.circle, new_circle):
                    a = k
                    a.next = b
                    b.previous = a
                    i -= 1
                    placed = True
                    break
                sk += k.circle.r
                k = k.previous
            
            if j == k.next:
                break
        
        if placed:
            i += 1
            continue
        
        # 成功找到位置，插入新节点
        new_node.previous = a
        new_node.next = b
        a.next = new_node
        b.previous = new_node
        b = new_node
        
        # 寻找新的最佳起始点
        aa = node_score(a)
        current = a
        while True:
            current = current.next
            if current == b:
                break
            ca = node_score(current)
            if ca < aa:
                a = current
                aa = ca
        
        b = a.next
        i += 1
    
    # 收集所有圆计算包围圆
    all_circles = []
    start = b
    current = start
    
    while True:
        all_circles.append(current.circle)
        current = current.next
        if current == start:
            break
    
    enclosing = find_enclosing_circle(all_circles, random_gen)
    
    # 平移所有圆
    for circle in circles:
        circle.x -= enclosing.x
        circle.y -= enclosing.y
    
    return enclosing.r

def visualize_circles(circles: List[Circle], title: str = ""):
    """可视化圆的位置（需要matplotlib）"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制所有圆
        for i, circle in enumerate(circles):
            patch = patches.Circle(
                (circle.x, circle.y), 
                circle.r, 
                fill=True, 
                alpha=0.5,
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(patch)
            
            # 标注半径
            ax.text(circle.x, circle.y, f'{circle.r:.1f}', 
                   ha='center', va='center', fontsize=8)
        
        # 设置坐标轴
        ax.set_aspect('equal')
        
        # 计算边界
        max_x = max(abs(c.x) + c.r for c in circles)
        max_y = max(abs(c.y) + c.r for c in circles)
        max_range = max(max_x, max_y) * 1.1
        
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{title} - {len(circles)} circles')
        ax.grid(True, alpha=0.3)
        
        plt.show()
        
    except ImportError:
        print("需要安装matplotlib进行可视化")
        print("运行: pip install matplotlib")

def main():
    """测试函数"""
    # 测试数据
    circles = [
        Circle(r=10),
        Circle(r=20),
        Circle(r=15),
        Circle(r=25),
        Circle(r=12),
        Circle(r=18),
        Circle(r=30)
    ]
    
    print("原始圆半径：")
    for i, circle in enumerate(circles):
        print(f"圆{i}: 半径={circle.r}")
    print()
    
    # 第一次打包
    print("=== 第一次打包（种子：12345） ===")
    gen1 = RandomGenerator(12345)
    circles_copy1 = [Circle(r=c.r) for c in circles]
    bound1 = pack_siblings_random(circles_copy1, gen1)  # 打包
    
    for i, circle in enumerate(circles_copy1):
        print(f"圆{i}: 半径={circle.r:.2f}, 位置=({circle.x:.6f}, {circle.y:.6f})")
    print(f"包围圆半径: {bound1:.6f}\n")
    
    # 可视化第一次结果
    visualize_circles(circles_copy1, "First packing")
    
    # 第二次打包（不同随机种子）
    print("=== 第二次打包（种子：67890） ===")
    gen2 = RandomGenerator(67890)
    circles_copy2 = [Circle(r=c.r) for c in circles]
    bound2 = pack_siblings_random(circles_copy2, gen2)
    
    for i, circle in enumerate(circles_copy2):
        print(f"圆{i}: 半径={circle.r:.2f}, 位置=({circle.x:.6f}, {circle.y:.6f})")
    print(f"包围圆半径: {bound2:.6f}")
    
    # 可视化第二次结果
    visualize_circles(circles_copy2, "Second packing")
    
    # 随机测试
    print("\n=== 随机圆测试 ===")
    random_circles = [Circle(r=random.uniform(5, 30)) for _ in range(15)]
    random_circles.sort(key=lambda c: c.r, reverse=True)  # 按半径降序排列
    
    gen3 = RandomGenerator(int(time.time()))
    bound3 = pack_siblings_random(random_circles, gen3)
    
    print(f"打包了 {len(random_circles)} 个随机圆")
    print(f"包围圆半径: {bound3:.6f}")
    
    visualize_circles(random_circles, "Random packing")

if __name__ == "__main__":
    import time
    main()