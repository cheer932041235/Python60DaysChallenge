# main.py
from circle import calculate_area

radius = 5
area = calculate_area(radius)
print(f"半径为 {radius} 的圆，面积是: {area}")

# 或者注释上面的，用下面的方法
# import circle 
# radius = 5
# area = circle.calculate_area(radius)
# print(f"半径为 {radius} 的圆，面积是: {area}")
