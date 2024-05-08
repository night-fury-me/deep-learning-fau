import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        if self.resolution % (2 * self.tile_size) != 0:
            self.output = np.empty(0)
            return self.output.copy()
        
        black_tile = np.zeros((self.tile_size, self.tile_size), dtype=np.int8)
        
        if(self.resolution == self.tile_size):
            self.output = black_tile
            return self.output.copy()

        white_tile = np.ones((self.tile_size, self.tile_size), dtype=np.int8)

        composite_tile_1 = np.hstack((black_tile, white_tile))
        composite_tile_2 = np.hstack((white_tile, black_tile))
        
        composite_tile = np.vstack((composite_tile_1, composite_tile_2))

        n_repeatation = self.resolution // (2 * self.tile_size)
        self.output = np.tile(composite_tile, (n_repeatation, n_repeatation))
        return self.output.copy()

    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='grey')
        plt.axis('off')  
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.center_X, self.center_Y = position
        self.output = None

    def draw(self):
        lin_space_X = np.linspace(0, self.resolution - 1, self.resolution)
        lin_space_Y = np.linspace(0, self.resolution - 1, self.resolution)

        X_cords, Y_cords = np.meshgrid(lin_space_X, lin_space_Y)
        distance_wrt_center = np.sqrt((X_cords - self.center_X)**2 + (Y_cords - self.center_Y)**2)

        self.output = distance_wrt_center <= self.radius

        return self.output.copy()
    
    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output, cmap='grey')
        plt.axis('off') 
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None
        
    def draw(self):
        color_X = np.linspace(0, 1, self.resolution).reshape(1, -1)
        color_Y = np.linspace(0, 1, self.resolution).reshape(-1, 1)

        red_channel = np.vstack([color_X] * self.resolution)
        green_channel = np.hstack([color_Y] * self.resolution)
        blue_channel = np.vstack([color_X[:, ::-1]] * self.resolution)
        
        self.output = np.stack((red_channel, green_channel, blue_channel), axis = -1)
        return self.output.copy()
        
    def show(self):
        if self.output is None:
            self.draw()
        plt.imshow(self.output)
        plt.axis('off') 
        plt.show()
