# deepseek深度思考版本
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import random
import math


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Circle:
    """圆类，存储圆心坐标和半径"""
    def __init__(self, x: float = 0.0, y: float = 0.0, r: float = 0.0):
        self.x = x
        self.y = y
        self.r = r
    
    def __repr__(self):
        return f"Circle(x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f})"


def place_circle(a: Circle, b: Circle, c: Circle) -> None:
    """根据两个已知圆a和b，计算第三个圆c的位置（外切）"""
    dx = b.x - a.x
    dy = b.y - a.y
    d2 = dx*dx + dy*dy
    
    if d2 > 0:
        a2 = a.r + c.r
        a2 *= a2
        b2 = b.r + c.r
        b2 *= b2
        
        if a2 > b2:
            x = (d2 + b2 - a2) / (2 * d2)
            y = math.sqrt(max(0, b2 / d2 - x*x))
            c.x = b.x - x*dx - y*dy
            c.y = b.y - x*dy + y*dx
        else:
            x = (d2 + a2 - b2) / (2 * d2)
            y = math.sqrt(max(0, a2 / d2 - x*x))
            c.x = a.x + x*dx - y*dy
            c.y = a.y + x*dy + y*dx
    else:
        c.x = a.x + c.r
        c.y = a.y


def circles_intersect(a: Circle, b: Circle) -> bool:
    """判断两个圆是否相交（含微小容差）"""
    dx = b.x - a.x
    dy = b.y - a.y
    dr = a.r + b.r - 1e-6
    return dr > 0 and dr*dr > dx*dx + dy*dy


class Node:
    """双向链表节点，用于表示圆环链"""
    def __init__(self, circle: Circle):
        self.circle = circle
        self.next: Optional['Node'] = None
        self.prev: Optional['Node'] = None


def pack_circles_siblings(circles: List[Circle], seed: Optional[int] = None) -> float:
    """
    圆打包算法（类似d3.layout.pack的siblings算法）
    返回包围所有圆的最小圆的半径
    """
    if not circles:
        return 0.0
    
    n = len(circles)
    
    # 按半径降序排序（通常可以得到更好的打包效果）
    circles.sort(key=lambda c: c.r, reverse=True)
    
    # 放置第一个圆
    a = circles[0]
    a.x = 0.0
    a.y = 0.0
    
    if n == 1:
        return a.r
    
    # 放置第二个圆
    b = circles[1]
    a.x = -b.r
    b.x = a.r
    b.y = 0.0
    
    if n == 2:
        return a.r + b.r
    
    # 放置第三个圆
    c = circles[2]
    place_circle(b, a, c)
    
    # 创建双向链表
    a_node = Node(a)
    b_node = Node(b)
    c_node = Node(c)
    
    # 连接节点形成环
    a_node.next = c_node
    a_node.prev = b_node
    
    b_node.next = a_node
    b_node.prev = c_node
    
    c_node.next = b_node
    c_node.prev = a_node
    
    # 设置当前节点
    current_a = a_node
    current_b = b_node
    
    # 放置剩余圆
    for i in range(3, n):
        new_circle = circles[i]
        place_circle(current_a.circle, current_b.circle, new_circle)
        new_node = Node(new_circle)
        
        # 查找插入位置
        j = current_b.next
        k = current_a.prev
        sj = current_b.circle.r
        sk = current_a.circle.r
        
        while True:
            if sj <= sk:
                if j and circles_intersect(j.circle, new_circle):
                    current_b = j
                    current_a.next = current_b
                    current_b.prev = current_a
                    place_circle(current_a.circle, current_b.circle, new_circle)
                    new_node = Node(new_circle)
                    j = current_b.next
                    k = current_a.prev
                    sj = current_b.circle.r
                    sk = current_a.circle.r
                    continue
                if j:
                    sj += j.circle.r
                    j = j.next
            else:
                if k and circles_intersect(k.circle, new_circle):
                    current_a = k
                    current_a.next = current_b
                    current_b.prev = current_a
                    place_circle(current_a.circle, current_b.circle, new_circle)
                    new_node = Node(new_circle)
                    j = current_b.next
                    k = current_a.prev
                    sj = current_b.circle.r
                    sk = current_a.circle.r
                    continue
                if k:
                    sk += k.circle.r
                    k = k.prev
            
            # 如果已经检查完整个环
            if j == k.next if k else False:
                break
        
        # 插入新节点
        new_node.prev = current_a
        new_node.next = current_b
        current_a.next = new_node
        current_b.prev = new_node
        
        # 选择新的参考节点
        current_b = new_node
    
    # 计算包围所有圆的最小圆
    return compute_enclosing_circle(circles)


def compute_enclosing_circle(circles: List[Circle]) -> float:
    """计算包围所有圆的最小圆的半径"""
    if not circles:
        return 0.0
    
    # 简单的包围圆算法：先找到所有圆的边界，然后计算最小包围圆
    min_x = min(c.x - c.r for c in circles)
    max_x = max(c.x + c.r for c in circles)
    min_y = min(c.y - c.r for c in circles)
    max_y = max(c.y + c.r for c in circles)
    
    # 计算中心点
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # 计算最大距离（圆心到中心点的距离 + 半径）
    max_distance = 0.0
    for circle in circles:
        distance = math.hypot(circle.x - center_x, circle.y - center_y) + circle.r
        max_distance = max(max_distance, distance)
    
    # 平移所有圆，使中心点位于原点
    for circle in circles:
        circle.x -= center_x
        circle.y -= center_y
    
    return max_distance


def generate_random_circles(n: int, min_radius: float = 1.0, max_radius: float = 10.0) -> List[Circle]:
    """生成随机半径的圆"""
    return [Circle(r=random.uniform(min_radius, max_radius)) for _ in range(n)]


def generate_weighted_circles(n: int) -> List[Circle]:
    """生成具有不同权重的圆（半径不同）"""
    circles = []
    for i in range(n):
        # 半径随指数分布，使得少数圆较大，多数圆较小
        r = 5 + 20 * (1 - (i / n) ** 2)
        circles.append(Circle(r=r))
    return circles


def plot_circles(circles: List[Circle], title: str = "圆打包结果", show_axes: bool = False):
    """使用matplotlib绘制圆"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 设置坐标轴
    if not show_axes:
        ax.set_aspect('equal')
        ax.axis('off')
    
    # 计算边界
    max_r = max(c.r for c in circles) if circles else 0
    max_extent = 0
    for circle in circles:
        extent = max(abs(circle.x) + circle.r, abs(circle.y) + circle.r)
        max_extent = max(max_extent, extent)
    
    # 设置图形边界（增加一些边距）
    margin = max_r * 0.1
    limit = max_extent + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # 绘制每个圆
    colors = plt.cm.Set3(np.linspace(0, 1, len(circles)))
    
    for i, circle in enumerate(circles):
        # 创建圆形的补丁
        circle_patch = mpatches.Circle(
            (circle.x, circle.y), 
            circle.r, 
            facecolor=colors[i % len(colors)], 
            edgecolor='black',
            linewidth=1,
            alpha=0.7
        )
        ax.add_patch(circle_patch)
        
        # 可选：添加圆心标记
        ax.plot(circle.x, circle.y, 'k.', markersize=3)
        
        # 可选：添加半径标签
        ax.text(circle.x, circle.y, f'{circle.r:.1f}', 
                fontsize=8, ha='center', va='center')
    
    # 添加标题
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 添加统计信息
    total_circles = len(circles)
    total_area = sum(math.pi * c.r**2 for c in circles)
    enclosing_radius = max(math.hypot(c.x, c.y) + c.r for c in circles) if circles else 0
    packing_efficiency = total_area / (math.pi * enclosing_radius**2) if enclosing_radius > 0 else 0
    
    info_text = f"圆的数量: {total_circles}\n"
    info_text += f"总面积: {total_area:.1f}\n"
    info_text += f"包围圆半径: {enclosing_radius:.1f}\n"
    info_text += f"填充效率: {packing_efficiency*100:.1f}%"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax


def run_example():
    """运行示例"""
    print("=== 圆打包算法演示 ===\n")
    
    # 示例1：随机圆
    print("示例1: 随机生成20个圆")
    circles1 = generate_random_circles(20, 1.0, 8.0)
    radius1 = pack_circles_siblings(circles1)
    print(f"包围圆半径: {radius1:.2f}")
    
    fig1, ax1 = plot_circles(circles1, "随机圆打包结果")
    
    # 示例2：加权圆（少数大圆，多数小圆）
    print("\n示例2: 加权圆（少数大圆，多数小圆）")
    circles2 = generate_weighted_circles(30)
    radius2 = pack_circles_siblings(circles2)
    print(f"包围圆半径: {radius2:.2f}")
    
    fig2, ax2 = plot_circles(circles2, "加权圆打包结果")
    
    # 示例3：相同半径的圆（最密集情况）
    print("\n示例3: 相同半径的圆")
    circles3 = [Circle(r=5.0) for _ in range(15)]
    radius3 = pack_circles_siblings(circles3)
    print(f"包围圆半径: {radius3:.2f}")
    
    fig3, ax3 = plot_circles(circles3, "相同半径圆打包结果")
    
    plt.show()
    
    # 返回图形对象以便进一步操作
    return [fig1, fig2, fig3]


def interactive_packing():
    """交互式圆打包演示"""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    if 'get_ipython' not in globals():
        print("此功能需要在Jupyter Notebook中运行")
        return
    
    # 创建控件
    n_circles = widgets.IntSlider(
        value=15, min=5, max=50, step=1,
        description='圆的数量:',
        style={'description_width': 'initial'}
    )
    
    max_radius = widgets.FloatSlider(
        value=8.0, min=1.0, max=20.0, step=0.5,
        description='最大半径:',
        style={'description_width': 'initial'}
    )
    
    min_radius = widgets.FloatSlider(
        value=1.0, min=0.5, max=10.0, step=0.5,
        description='最小半径:',
        style={'description_width': 'initial'}
    )
    
    random_seed = widgets.IntSlider(
        value=42, min=0, max=100, step=1,
        description='随机种子:',
        style={'description_width': 'initial'}
    )
    
    generate_button = widgets.Button(description='生成并打包')
    output = widgets.Output()
    
    def on_generate_clicked(b):
        with output:
            clear_output(wait=True)
            
            # 设置随机种子
            random.seed(random_seed.value)
            np.random.seed(random_seed.value)
            
            # 生成圆
            circles = generate_random_circles(
                n_circles.value, 
                min_radius.value, 
                max_radius.value
            )
            
            # 执行打包算法
            enclosing_radius = pack_circles_siblings(circles)
            
            # 绘制结果
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_circles(circles, f"圆打包结果 (n={n_circles.value})")
            plt.show()
            
            # 打印统计信息
            print(f"圆数量: {n_circles.value}")
            print(f"半径范围: {min_radius.value:.1f} - {max_radius.value:.1f}")
            print(f"包围圆半径: {enclosing_radius:.2f}")
            
            total_area = sum(math.pi * c.r**2 for c in circles)
            packing_efficiency = total_area / (math.pi * enclosing_radius**2)
            print(f"填充效率: {packing_efficiency*100:.1f}%")
    
    generate_button.on_click(on_generate_clicked)
    
    # 显示控件
    display(widgets.VBox([
        widgets.HBox([n_circles, min_radius]),
        widgets.HBox([max_radius, random_seed]),
        generate_button,
        output
    ]))


if __name__ == "__main__":
    # 运行示例
    figures = run_example()
    
    # 如果需要保存图像
    # for i, fig in enumerate(figures):
    #     fig.savefig(f'circle_packing_{i}.png', dpi=150, bbox_inches='tight')
    
    # 如果想要在Jupyter Notebook中运行交互式演示，取消下面的注释
    # interactive_packing()


# deepseek 优化
import math
import random
from typing import List, Dict, Any


def place(b: Dict[str, float], a: Dict[str, float], c: Dict[str, float]) -> None:
    """根据两个已知圆a和b，计算第三个圆c的位置（外切）"""
    dx = b['x'] - a['x']
    dy = b['y'] - a['y']
    d2 = dx * dx + dy * dy
    
    if d2:
        a2 = a['r'] + c['r']
        a2 *= a2
        b2 = b['r'] + c['r']
        b2 *= b2
        
        if a2 > b2:
            x = (d2 + b2 - a2) / (2 * d2)
            y = math.sqrt(max(0, b2 / d2 - x * x))
            c['x'] = b['x'] - x * dx - y * dy
            c['y'] = b['y'] - x * dy + y * dx
        else:
            x = (d2 + a2 - b2) / (2 * d2)
            y = math.sqrt(max(0, a2 / d2 - x * x))
            c['x'] = a['x'] + x * dx - y * dy
            c['y'] = a['y'] + x * dy + y * dx
    else:
        c['x'] = a['x'] + c['r']
        c['y'] = a['y']


def intersects(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """判断两个圆是否相交（含微小容差）"""
    dr = a['r'] + b['r'] - 1e-6
    dx = b['x'] - a['x']
    dy = b['y'] - a['y']
    return dr > 0 and dr * dr > dx * dx + dy * dy


def score(node: 'Node') -> float:
    """计算节点与下一节点组合的分数"""
    a = node.circle
    b = node.next.circle
    ab = a['r'] + b['r']
    dx = (a['x'] * b['r'] + b['x'] * a['r']) / ab
    dy = (a['y'] * b['r'] + b['y'] * a['r']) / ab
    return dx * dx + dy * dy


class Node:
    """双向链表节点"""
    def __init__(self, circle: Dict[str, float]):
        self.circle = circle
        self.next = None
        self.prev = None


def pack_enclosing_circle(circles: List[Dict[str, float]]) -> Dict[str, float]:
    """计算包围所有圆的最小圆（简化版本）"""
    if not circles:
        return {'x': 0, 'y': 0, 'r': 0}
    
    # 计算所有圆的边界
    min_x = min(c['x'] - c['r'] for c in circles)
    max_x = max(c['x'] + c['r'] for c in circles)
    min_y = min(c['y'] - c['r'] for c in circles)
    max_y = max(c['y'] + c['r'] for c in circles)
    
    # 计算中心点
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    
    # 计算半径（最远距离）
    max_distance = 0
    for circle in circles:
        distance = math.hypot(circle['x'] - cx, circle['y'] - cy) + circle['r']
        if distance > max_distance:
            max_distance = distance
    
    return {'x': cx, 'y': cy, 'r': max_distance}


def pack_siblings_random(circles: List[Dict[str, float]]) -> float:
    """原JS算法的Python实现"""
    n = len(circles)
    if n == 0:
        return 0.0
    
    # 按照原算法，不排序
    # 第一个圆
    a = circles[0]
    a['x'] = 0.0
    a['y'] = 0.0
    
    if n == 1:
        return a['r']
    
    # 第二个圆
    b = circles[1]
    a['x'] = -b['r']
    b['x'] = a['r']
    b['y'] = 0.0
    
    if n == 2:
        return a['r'] + b['r']
    
    # 第三个圆
    c = circles[2]
    place(b, a, c)
    
    # 初始化双向循环链表
    a_node = Node(a)
    b_node = Node(b)
    c_node = Node(c)
    
    a_node.next = b_node
    a_node.prev = c_node
    
    b_node.next = c_node
    b_node.prev = a_node
    
    c_node.next = a_node
    c_node.prev = b_node
    
    a = a_node
    b = b_node
    
    # 放置剩余圆
    i = 3
    while i < n:
        circle = circles[i]
        place(b.circle, a.circle, circle)
        c_node = Node(circle)
        
        # 查找最近的相交圆
        j = b.next
        k = a.prev
        sj = b.circle['r']
        sk = a.circle['r']
        
        found_intersection = False
        while True:
            if sj <= sk:
                if intersects(j.circle, c_node.circle):
                    b = j
                    a.next = b
                    b.prev = a
                    found_intersection = True
                    break
                sj += j.circle['r']
                j = j.next
            else:
                if intersects(k.circle, c_node.circle):
                    a = k
                    a.next = b
                    b.prev = a
                    found_intersection = True
                    break
                sk += k.circle['r']
                k = k.prev
            
            if j == k.next:
                break
        
        if found_intersection:
            # 重新尝试放置当前圆
            continue
        
        # 插入新节点
        c_node.prev = a
        c_node.next = b
        a.next = c_node
        b.prev = c_node
        b = c_node
        
        # 找到距离质心最近的节点
        aa = score(a)
        current = a.next
        while current != b:
            ca = score(current)
            if ca < aa:
                a = current
                aa = ca
            current = current.next
        
        b = a.next
        i += 1
    
    # 计算包围圆
    chain_circles = [b.circle]
    current = b.next
    while current != b:
        chain_circles.append(current.circle)
        current = current.next
    
    enclosing_circle = pack_enclosing_circle(chain_circles)
    
    # 平移所有圆使包围圆居中于原点
    for circle in circles:
        circle['x'] -= enclosing_circle['x']
        circle['y'] -= enclosing_circle['y']
    
    return enclosing_circle['r']


def create_random_circles(n: int, min_r: float = 1.0, max_r: float = 10.0) -> List[Dict[str, float]]:
    """创建随机圆"""
    circles = []
    for i in range(n):
        r = random.uniform(min_r, max_r)
        circles.append({'x': 0.0, 'y': 0.0, 'r': r, 'id': i})
    return circles


def create_weighted_circles(n: int) -> List[Dict[str, float]]:
    """创建加权圆"""
    circles = []
    for i in range(n):
        # 少数大圆，多数小圆
        if i < n // 10:  # 前10%是大圆
            r = 5 + 20 * (1 - i / (n // 10))
        else:
            r = random.uniform(1.0, 5.0)
        circles.append({'x': 0.0, 'y': 0.0, 'r': r, 'id': i})
    return circles


def check_overlaps(circles: List[Dict[str, float]]) -> List[tuple]:
    """检查圆之间的重叠"""
    overlaps = []
    n = len(circles)
    
    for i in range(n):
        for j in range(i + 1, n):
            c1 = circles[i]
            c2 = circles[j]
            distance = math.hypot(c1['x'] - c2['x'], c1['y'] - c2['y'])
            min_distance = c1['r'] + c2['r'] - 0.01  # 微小容差
            
            if distance < min_distance:
                overlap_amount = min_distance - distance
                overlaps.append((i, j, overlap_amount))
    
    return overlaps


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_circles_js_algorithm(circles: List[Dict[str, float]], title: str = "JS算法打包结果"):
    """绘制圆（使用JS算法）"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 计算边界
    max_r = max(c['r'] for c in circles) if circles else 0
    max_extent = 0
    for circle in circles:
        extent = max(abs(circle['x']) + circle['r'], abs(circle['y']) + circle['r'])
        max_extent = max(max_extent, extent)
    
    # 设置图形边界
    margin = max_r * 0.1
    limit = max_extent + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # 检查重叠
    overlaps = check_overlaps(circles)
    
    # 绘制每个圆
    colors = plt.cm.Set3(range(len(circles)))
    
    for i, circle in enumerate(circles):
        # 检查是否有重叠
        has_overlap = any(i in overlap[:2] for overlap in overlaps)
        color = 'red' if has_overlap else colors[i % len(colors)]
        alpha = 0.5 if has_overlap else 0.7
        
        # 创建圆形补丁
        circle_patch = mpatches.Circle(
            (circle['x'], circle['y']), 
            circle['r'], 
            facecolor=color, 
            edgecolor='black',
            linewidth=1,
            alpha=alpha
        )
        ax.add_patch(circle_patch)
        
        # 添加圆心标记
        # ax.plot(circle['x'], circle['y'], 'k.', markersize=3)
        
        # 添加标签
        label = f"{circle['id']}\nr={circle['r']:.1f}"
        ax.text(circle['x'], circle['y'], label
                )
    
    # 绘制重叠线
    for i, j, amount in overlaps:
        c1, c2 = circles[i], circles[j]
        ax.plot([c1['x'], c2['x']], [c1['y'], c2['y']], 'r-', linewidth=1, alpha=0.5)
    
    # 添加标题和统计信息
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    total_circles = len(circles)
    total_area = sum(math.pi * c['r']**2 for c in circles)
    enclosing_radius = max(math.hypot(c['x'], c['y']) + c['r'] for c in circles) if circles else 0
    packing_efficiency = total_area / (math.pi * enclosing_radius**2) if enclosing_radius > 0 else 0
    
    info_text = f"圆的数量: {total_circles}\n"
    info_text += f"总面积: {total_area:.1f}\n"
    info_text += f"估计包围半径: {enclosing_radius:.1f}\n"
    info_text += f"填充效率: {packing_efficiency*100:.1f}%\n"
    info_text += f"重叠对数: {len(overlaps)}"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    return fig, ax


def test_js_algorithm():
    """测试JS算法"""
    print("=== 测试JS圆打包算法 ===\n")
    
    # 设置随机种子以便重现
    random.seed(42)
    
    # 测试1: 随机圆
    print("测试1: 15个随机圆")
    circles1 = create_random_circles(15, 1.0, 8.0)
    radius1 = pack_siblings_random(circles1)
    print(f"返回的包围圆半径: {radius1:.2f}")
    
    overlaps1 = check_overlaps(circles1)
    print(f"重叠对数: {len(overlaps1)}")
    if overlaps1:
        for i, j, amount in overlaps1:
            print(f"  圆{circles1[i]['id']}和圆{circles1[j]['id']}重叠: {amount:.4f}")
    
    fig1, ax1 = plot_circles_js_algorithm(circles1, "JS算法 - 随机圆")
    
    # 测试2: 加权圆
    print("\n测试2: 30个加权圆")
    circles2 = create_weighted_circles(30)
    radius2 = pack_siblings_random(circles2)
    print(f"返回的包围圆半径: {radius2:.2f}")
    
    overlaps2 = check_overlaps(circles2)
    print(f"重叠对数: {len(overlaps2)}")
    if overlaps2:
        for i, j, amount in overlaps2[:5]:  # 只显示前5个
            print(f"  圆{circles2[i]['id']}和圆{circles2[j]['id']}重叠: {amount:.4f}")
        if len(overlaps2) > 5:
            print(f"  ... 还有{len(overlaps2)-5}个重叠")
    
    fig2, ax2 = plot_circles_js_algorithm(circles2, "JS算法 - 加权圆")
    
    # 测试3: 相同半径的圆
    print("\n测试3: 10个相同半径的圆")
    circles3 = [{'x': 0.0, 'y': 0.0, 'r': 5.0, 'id': i} for i in range(10)]
    radius3 = pack_siblings_random(circles3)
    print(f"返回的包围圆半径: {radius3:.2f}")
    
    overlaps3 = check_overlaps(circles3)
    print(f"重叠对数: {len(overlaps3)}")
    
    fig3, ax3 = plot_circles_js_algorithm(circles3, "JS算法 - 相同半径圆")
    
    plt.show()
    
    return circles1, circles2, circles3


def compare_with_sorted():
    """比较排序和不排序的效果"""
    print("\n=== 比较排序对算法的影响 ===")
    
    # 创建相同的圆集
    circles_unsorted = create_weighted_circles(20)
    circles_sorted = sorted(circles_unsorted.copy(), key=lambda c: c['r'], reverse=True)
    
    # 运行算法（不排序）
    radius_unsorted = pack_siblings_random(circles_unsorted)
    overlaps_unsorted = check_overlaps(circles_unsorted)
    
    # 运行算法（排序）
    radius_sorted = pack_siblings_random(circles_sorted)
    overlaps_sorted = check_overlaps(circles_sorted)
    
    print(f"不排序 - 包围半径: {radius_unsorted:.2f}, 重叠对数: {len(overlaps_unsorted)}")
    print(f"排序后 - 包围半径: {radius_sorted:.2f}, 重叠对数: {len(overlaps_sorted)}")
    
    # 绘图比较
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 不排序的结果
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("JS算法 - 不排序", fontsize=12)
    
    # 排序的结果
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title("JS算法 - 按半径降序排序", fontsize=12)
    
    # 绘制两个子图
    for ax, circles, title in [(ax1, circles_unsorted, "不排序"), (ax2, circles_sorted, "排序后")]:
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 计算边界
        max_r = max(c['r'] for c in circles)
        max_extent = max(max(abs(c['x']) + c['r'], abs(c['y']) + c['r']) for c in circles)
        limit = max_extent + max_r * 0.1
        
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        # 绘制圆
        colors = plt.cm.Set3(range(len(circles)))
        for i, circle in enumerate(circles):
            circle_patch = mpatches.Circle(
                (circle['x'], circle['y']), 
                circle['r'], 
                facecolor=colors[i % len(colors)], 
                edgecolor='black',
                linewidth=1,
                alpha=0.7
            )
            ax.add_patch(circle_patch)
            ax.plot(circle['x'], circle['y'], 'k.', markersize=3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 运行JS算法测试
    circles1, circles2, circles3 = test_js_algorithm()
    
    # 比较排序效果
    compare_with_sorted()
    
    print("\n=== 分析 ===")
    print("JS算法在放置圆时，使用链表结构追踪外部边界链(front-chain)")
    print("算法尝试将新圆放在链上两个相邻圆之间，但如果与链上其他圆相交，会调整链")
    print("这种方法的优点是速度快，但可能产生重叠，特别是在圆大小差异较大时")
    print("原算法没有对输入圆排序，所以结果可能不稳定")
