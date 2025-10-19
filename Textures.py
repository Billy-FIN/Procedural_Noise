from cgitb import text
import sys
import numpy as np
import cv2
import math
import random
random.seed(9814072356) #KEEP THIS VALUE FOR GRADING PURPOSES (comment it out if you want to test other values but uncomment it for the final submission)

#this is just a helper function if you want it
def distance(a,b):
    return math.sqrt(a*a + b*b)

def WhiteNoise(texture):
    #TODO 1
    #use the python random() to fill the texture with random values from 0 to 1
    #be sure to scale to 0 to 255 for the final texture
    ########################################YOUR CODE HERE########################################
    for i in range(len(texture)):
        for j in range(len(texture[i])):
            texture[i][j] = random.random() * 255.0
    ##############################################################################################
    return

def lerp(a, b, w):
    #TODO 2
    #linearly interpolate between two values given a value from 0 to 1
    lin = 0.0
    ########################################YOUR CODE HERE########################################
    lin = (1 - w) * a + w * b
    ##############################################################################################
    return lin

def slerp(a, b, w):
    #This is NOT a TO DO, its an extra interpolation funciton for you to play with
    #this is a 5th degree interpolation where the derivative and 2nd degree derivative are smooth
    lin = 0.0
    lin = (b - a) * ((w * (w * 6.0 - 15.0) + 10.0) * w * w * w) + a;
    return lin

def ValueNoise(texture):
    #TODO 3
    #use grid of random values (*cough* white noise *cough*) like in white noise to make value noise
    random_values = np.zeros(shape=(50,50)) #one extra row and collumn for interpolation
    #fill the texture with bilinearly interpolated values from the grid https://en.wikipedia.org/wiki/Bilinear_interpolation
    # we need to know how far along the values the current pixel is,
    #convert the pixel coords to value coords and a residual using math.modf()
    ########################################YOUR CODE HERE########################################
    for i in range(len(random_values)):
        for j in range(len(random_values[i])):
            random_values[i][j] = random.random() * 255.0
    for i in range(len(texture)):
        for j in range(len(texture[i])):
            row_weight, grid_point_row = math.modf((i / len(texture)) * 49)
            col_weight, grid_point_col = math.modf((j / len(texture[i])) * 49)

            grid_point_row = int(grid_point_row)
            grid_point_col = int(grid_point_col)
            
            top_left = random_values[grid_point_row][grid_point_col]
            top_right = random_values[grid_point_row][grid_point_col + 1]
            bottom_left = random_values[grid_point_row + 1][grid_point_col]
            bottom_right = random_values[grid_point_row + 1][grid_point_col + 1]

            top = lerp(top_left, top_right, col_weight)
            bottom = lerp(bottom_left, bottom_right, col_weight)
            texture[i][j] = lerp(top, bottom, row_weight)
    ##############################################################################################
    return

def Grad2D(x,y):
    #TODO 4
    grid = np.zeros(shape=(x,y,2))
    #fill the above 3d array with a 2d array of random 2d unit-vectors
    #its like a 2d image where theres an x color and a y color from -1.0 to 1.0 that follows pythagorean theorem where the hypotenuse is 1
    ########################################YOUR CODE HERE########################################
    for i in range(x):
        for j in range(y):
            X = random.random()
            Y = math.sqrt(1 - X * X) 
            if random.random() > 0.5:
                X = -X
            if random.random() > 0.5:
                Y = -Y
            grid[i][j][0] = X
            grid[i][j][1] = Y
    ##############################################################################################
    return grid

def PerlinNoise(texture, sizex = 50, sizey = 50, amp = 255.0):
    #TODO 5
    #wikipedia has a more advanced 2d implementation that does this per-pixel you can look at 
    #but basing this off a version from perlin's thesis (whole texture at once)
    #other parameters are used for fractal noise later
    #were gonna break this into parts.
    #Get a random grid to sample from
    g2d = Grad2D(sizex,sizey)
    for i in range(len(texture)):
        for j in range(len(texture[i])):
    ########################################YOUR CODE HERE########################################
            #for each point calculate the dot product of its vector and the vector from that vectors position on the grid
            #1. first get the coords the same way as in value noise
            row_weight, grid_point_row = math.modf((i / len(texture)) * (sizex - 1))
            col_weight, grid_point_col = math.modf((j / len(texture[i])) * (sizey - 1))

            grid_point_row = int(grid_point_row)
            grid_point_col = int(grid_point_col)

            #2.next use those components to index the grid for the vector from the grid point to the pixel and compute the dot product
            #Do this for all four surrounding grid points. the wikipedia has a helpful graphic showing this
            top_left_grad = g2d[grid_point_row][grid_point_col]
            top_right_grad = g2d[grid_point_row][grid_point_col + 1]
            bottom_left_grad = g2d[grid_point_row + 1][grid_point_col]
            bottom_right_grad = g2d[grid_point_row + 1][grid_point_col + 1]

            dist_to_top_left = np.array([row_weight, col_weight])
            dist_to_top_right = np.array([row_weight, col_weight - 1])
            dist_to_bottom_left = np.array([row_weight - 1, col_weight])
            dist_to_bottom_right = np.array([row_weight - 1, col_weight - 1])

            dot_top_left = np.dot(top_left_grad, dist_to_top_left)
            dot_top_right = np.dot(top_right_grad, dist_to_top_right)
            dot_bottom_left = np.dot(bottom_left_grad, dist_to_bottom_left)
            dot_bottom_right = np.dot(bottom_right_grad, dist_to_bottom_right)

            #3.finally, just like in value noise, bilineraly interpolate between the four dot products and assign to texture
            top = lerp(dot_top_left, dot_top_right, col_weight)
            bottom = lerp(dot_bottom_left, dot_bottom_right, col_weight)
            value = lerp(top, bottom, row_weight)
            
            #be sure to scale the output to between 0.0 and 1.0 then multiply by the amplitude to obtain the pixel value
            value = (value + 1) / 2 * amp
            texture[i][j] = value
    ##############################################################################################
    return

def FractalNoise(texture):
    #TODO 6
    #Fractal noise simply takes a noise function and generates it multiple times while scaling the inputs and outputs
    #In more detail: we use 4 variables
    amplitude = 80.0 #scale of this noise output
    frequency = 1.0 #inverse scale of the input
    gain = 0.8 #rate at which the amplitude is adjusted per iteration (non-linear)
    lacunarity = 0.6 #rate at which the frequency grows (similar to gain)
    #we scale the size of the output values by the amplitude and scale the gradient grid by the frequency
    #we then scale the amp and freq by the gain and lacunarity to get ready for the next iteration
    octaves = 6 #how many iterations (layers of noise) do we want?
    #use the Perlin noise function here but you could in theory jury-rig the other funcitons to work fractally as well
    texture_layer = np.zeros(texture.shape, dtype=np.uint8) #temp texture to hold one layer of noise
    ########################################YOUR CODE HERE########################################
    for _ in range(octaves):
        PerlinNoise(texture_layer, int(50 * frequency), int(50 * frequency), amplitude)
        texture += texture_layer
        amplitude *= gain
        frequency /= lacunarity
    ##############################################################################################
    return

def WorleyNoise(texture, cells_x = 10, cells_y = 10): #increase cell count at your own risk
    #TODO 7
    feature_points = np.zeros(shape=(cells_x,cells_y,2), dtype=np.uint32)
    shape = texture.shape
    print(shape[0]//cells_x)
    print(shape[1]//cells_y)
    #Fill the feature_points with a random point in each of the cells_x*cells_y cells
    ########################################YOUR CODE HERE########################################
    cell_width = shape[0] // cells_x
    cell_height = shape[1] // cells_y
    
    for i in range(cells_x):
        for j in range(cells_y):
            x_offset = random.random() * cell_width
            y_offset = random.random() * cell_height
            feature_points[i][j][0] = i * cell_width + x_offset
            feature_points[i][j][1] = j * cell_height + y_offset
    ##############################################################################################

    #we assume the nearest feature point is within one cell away (it usually is) so we only check the cell and its immediate neighbors (9 cells)
    #set each pixel value to the distance to the nearest feature point
    ########################################YOUR CODE HERE########################################
    for i in range(shape[0]):
        for j in range(shape[1]):
            min_distance = float('inf')

            current_cell_x = i // cell_width
            current_cell_y = j // cell_height

            for x_neighbor in [-1, 0, 1]:
                for y_neighbor in [-1, 0, 1]:
                    neighbor_cell_x = current_cell_x + x_neighbor
                    neighbor_cell_y = current_cell_y + y_neighbor

                    if 0 <= neighbor_cell_x < cells_x and 0 <= neighbor_cell_y < cells_y:
                        feature_point = feature_points[neighbor_cell_x][neighbor_cell_y]
                        dist = distance(i - int(feature_point[0]), j - int(feature_point[1]))
                        if dist < min_distance:
                            min_distance = dist
            texture[i][j] = min_distance
    ##############################################################################################
    return

def Art(texture):
    #TODO 8
    #make sumthin' cool lookin'

    height, width, rgb = texture.shape
    perlin_layer = np.zeros(shape=(height, width), dtype=np.uint8)
    fractal_layer = np.zeros(shape=(height, width), dtype=np.uint8)
    worley_layer = np.zeros(shape=(height, width), dtype=np.uint8)

    PerlinNoise(perlin_layer)
    FractalNoise(fractal_layer)
    WorleyNoise(worley_layer)

    texture[:,:,0] = perlin_layer
    texture[:,:,1] = fractal_layer
    texture[:,:,2] = worley_layer

    return

if __name__ == "__main__":
    #avoid making changes to this
    if len(sys.argv) > 1:
        if sys.argv[1] == "white":
            white_noise_texture = np.zeros(shape=(1000, 1000), dtype=np.uint8)
            WhiteNoise(white_noise_texture)
            cv2.imwrite("WhiteNoise.png",white_noise_texture)
        elif sys.argv[1] == "value":
            white_noise_texture = np.zeros(shape=(1000, 1000), dtype=np.uint8)
            ValueNoise(white_noise_texture)
            cv2.imwrite("ValueNoise.png",white_noise_texture)
        elif sys.argv[1] == "perlin":
            perlin_noise_texture = np.zeros(shape=(1000, 1000), dtype=np.uint8)
            PerlinNoise(perlin_noise_texture)
            cv2.imwrite("PerlinNoise.png",perlin_noise_texture)
        elif sys.argv[1] == "fractal":
            fractal_noise_texture = np.zeros(shape=(1000, 1000), dtype=np.uint8)
            FractalNoise(fractal_noise_texture)
            cv2.imwrite("FractalNoise.png",fractal_noise_texture)
        elif sys.argv[1] == "worley":
            worley_noise_texture = np.zeros(shape=(1000, 1000), dtype=np.uint8)
            WorleyNoise(worley_noise_texture)
            cv2.imwrite("WorleyNoise.png",worley_noise_texture)
        elif sys.argv[1] == "art":
            #feel free to mess with this signature
            art_texture = np.zeros(shape=(1000, 1000, 3), dtype=np.uint8)
            Art(art_texture)
            cv2.imwrite("Art.png",art_texture)
    else:
        print("missing/impropper arguements: usage \"python Textures.py <white/value/perlin/fractal/worley/art>\"")