import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  仅用于激活 3D 功能

# 1. 定义函数
def f(x, y):
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20

# 2. 构造网格
x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 5, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 3. 绘制 3D 曲面
fig = plt.figure(figsize=(12, 5))

# 3D 轴
ax3d = fig.add_subplot(121, projection='3d')
surf = ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='k', lw=0.1)
ax3d.set_title(r'$f(x,y)=\sin(x)\cos(y)+\sin(xy)+\frac{x^2+y^2}{20}$')
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('f(x, y)')
fig.colorbar(surf, ax=ax3d, shrink=0.6)

# 4. 绘制等高线
ax2d = fig.add_subplot(122)
contour = ax2d.contour(X, Y, Z, levels=20, cmap='viridis')
ax2d.clabel(contour, inline=True, fontsize=6)
ax2d.set_title('Contour plot')
ax2d.set_xlabel('x')
ax2d.set_ylabel('y')
ax2d.set_aspect('equal')

plt.tight_layout()
plt.show(