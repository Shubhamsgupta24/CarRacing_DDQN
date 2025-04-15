# Wall coordinates (as start and end points)
walls = [
    (12, 451, 15, 130),
    (15, 130, 61, 58),
    (61, 58, 149, 14),
    (149, 14, 382, 20),
    (382, 20, 549, 31),
    (549, 31, 636, 58),
    (636, 58, 678, 102),
    (678, 102, 669, 167),
    (669, 167, 600, 206),
    (600, 206, 507, 214),
    (507, 214, 422, 232),
    (422, 232, 375, 263),
    (375, 263, 379, 283),
    (379, 283, 454, 299),
    (454, 299, 613, 286),
    (613, 286, 684, 238),
    (684, 238, 752, 180),
    (752, 180, 862, 185),
    (862, 185, 958, 279),
    (958, 279, 953, 410),
    (953, 410, 925, 505),
    (925, 505, 804, 566),
    (804, 566, 150, 570),
    (150, 570, 46, 529),
    (46, 529, 12, 451),
    (104, 436, 96, 161),
    (96, 161, 122, 122),
    (122, 122, 199, 91),
    (199, 91, 376, 94),
    (376, 94, 469, 100),
    (469, 100, 539, 102),
    (539, 102, 585, 121),
    (585, 121, 585, 139),
    (585, 139, 454, 158),
    (454, 158, 352, 183),
    (352, 183, 293, 239),
    (293, 239, 294, 318),
    (294, 318, 361, 357),
    (361, 357, 490, 373),
    (490, 373, 671, 359),
    (671, 359, 752, 300),
    (752, 300, 812, 310),
    (812, 310, 854, 369),
    (854, 369, 854, 429),
    (854, 429, 754, 483),
    (754, 483, 192, 489),
    (192, 489, 104, 436)
]

# Goal coordinates (as start and end points)
goals = [
    (10,200,120,200),
    (0,100,120,150),
    (0,0,150,130),
    (120,0,170,120),
    (200,0,200,120),
    (270,0,270,110),
    (350,0,350,110),
    (450,0,450,110),
    (525,0,525,110),
    (600,0,550,130),
    (550,130,700,60),
    (550,130,700,130),
    (550,130,650,200),
    (550,130,570,240),
    (410,130,430,260),
    (430,260,300,350),
    (430,260,260,260),
    (430,260,280,180),
    (430,260,400,400),
    (550,260,570,400),
    (750,400,650,200),
    (750,400,800,160),
    (750,400,950,240),
    (750,400,980,440),
    (750,400,900,600),
    (750,460,750,600),
    (670,460,670,600),
    (590,460,590,600),
    (510,460,510,600),
    (430,460,430,600),
    (350,460,350,600),
    (280,460,278,600),
    (210,460,190,600),
    (80,600,175,440),
    (150,420,0,570),
    (0,450,130,400),
    (0,380,130,380)
]

import matplotlib.pyplot as plt

# Function to calculate intersection of two line segments
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate determinants
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None  # Lines are parallel or coincident

    # Calculate intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    # Check if the intersection point is within both line segments
    if (
        min(x1, x2) <= px <= max(x1, x2)
        and min(y1, y2) <= py <= max(y1, y2)
        and min(x3, x4) <= px <= max(x3, x4)
        and min(y3, y4) <= py <= max(y3, y4)
    ):
        return px, py
    return None

# Create a figure and axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
plt.gca().invert_yaxis()

# Plot walls as black lines
for wall in walls:
    ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color="black", linewidth=2)

# Plot goals as green lines
for goal in goals:
    ax.plot([goal[0], goal[2]], [goal[1], goal[3]], color="green", linewidth=2)

# Find and plot intersections with labels
intersection_points = []
for wall in walls:
    for goal in goals:
        intersection = line_intersection(wall, goal)
        if intersection:
            intersection_points.append(intersection)
            ax.plot(intersection[0], intersection[1], "ro")  # Mark intersection with red dot
            ax.text(
                intersection[0],
                intersection[1],
                f"({intersection[0]:.0f}, {intersection[1]:.0f})",
                color="blue",
                fontsize=8,
                ha="center",
                va="bottom"
            )  # Add label near the point

# Add grid and labels
ax.set_title("Wall and Goal Visualization with Intersection Points")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.grid(True)

# Set equal aspect ratio
ax.set_aspect("equal", adjustable="box")

# Show the plot
plt.show()
