import noise
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
shape = (1000, 1000)
scale = 200.0
octaves = 1
persistence = 0.5
lacunarity = 8.0

world = np.zeros(shape)
for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i / scale,
                                    j / scale,
                                    octaves=octaves,
                                    persistence=persistence,
                                    lacunarity=lacunarity,
                                    repeatx=1024,
                                    repeaty=1024,
                                    base=42)

# plt.imshow(world,cmap='terrain')

lin_x = np.linspace(0, 1, shape[0], endpoint=False)
lin_y = np.linspace(0, 1, shape[1], endpoint=False)
x, y = np.meshgrid(lin_x, lin_y)

fig = matplotlib.pyplot.figure()
ax = fig.add_subplot(111, projection="3d")
#cmap = plt.get_cmap('terrain')
#cmap.set_bad('black')
#world[100:500, 200:] = np.nan
#world = np.ma.masked_equal(world, np.nan)
ax.plot3D(0.5, 0.5, 0.7, 'ro', alpha=0.5)
ax.plot_surface(x, y, world, cmap='terrain', edgecolor='none')
# ax.plot3D(0.4, 0.5, 0.9,  marker=".", markersize=10, color="red")
# world2 = np.empty(shape)
# world2[:] = np.nan
# world2[500:600] = 0.6
# ax.plot_surface(x, y, world2, cmap='terrain')
plt.show()
