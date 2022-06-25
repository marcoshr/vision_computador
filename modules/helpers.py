import numpy as np


### PRIMERA FASE: Obtener el gradiente de cada píxel
def get_gradient_matrix(img):
    print(len(img))
    print(type(img))
    print(img.shape)

    m, n, c = img.shape
    print(type(m), m)
    print(type(n), n)

    # result = np.array((m, n), np.int64)
    # result = np.array([0 for x in range(m)] for y in range(n))
    result = np.zeros((m, n))
    ## NB: The border is cut off
    for i in range(1, m-1):
        for j in range(1, n-1):
            result[i, j] = get_gradient(img, i, j)
    return result

def get_gradient(img, i, j):
    BGRLeft = img[i, j-1]
    BGRRight = img[i, j+1]
    BGRUp = img[i-1, j]
    BGRDown = img[i+1, j]

    diffX = (BGRRight - BGRLeft) / 2
    diffY = (BGRDown - BGRUp) / 2

    gXX = diffX[2]*diffX[2] + diffX[1]*diffX[1] + diffX[0]*diffX[0]
    gYY = diffY[2]*diffY[2] + diffY[1]*diffY[1] + diffY[0]*diffY[0]
    gXY = diffX[2]*diffY[2] + diffX[1]*diffY[1] + diffX[0]*diffY[0]

    argument = 0.5 * np.arctan(2*gXY / (gXX-gYY))
    brackets = (gXX+gYY) + (gXX-gYY)*np.cos(argument) + 2*gXY*np.sin(2*argument)
    norm = np.sqrt(0.5 * (brackets))

    # print(f'norm: {norm}')
    # print(f'brackets: {brackets}')

    # return (norm, brackets)
    return norm



### SEGUNDA FASE: Obtener la división en células

def split_in_cells(gradient_matrix, cell_size):
    print('--- split_in_cells ---')
    print(f'gradient_matrix shape: {gradient_matrix.shape}')
    m, n = gradient_matrix.shape
    cell_rows = int(m / cell_size)
    cell_columns = int(n / cell_size)
    print(f'cell_rows: {cell_rows}')
    print(f'cell_columns: {cell_columns}')

    # result = np.array(cell_rows, cell_columns)
    result = np.zeros((cell_rows, cell_columns))

    for i in range(cell_rows):
        for j in range(cell_columns):
            cell_img = gradient_matrix[i*cell_size : (i+1)*cell_size,  j*cell_size : (j+1)*cell_size]
            print(cell_img.shape)
            print(cell_img)
            print(result[i, j].shape)
            result[i, j] = cell_img

    return result



### TERCERA FASE: Obtener histograma por celda

def getHistogram(cell):
    (m, n) = cell.shape()  # It should be m = n = cellSize
    categories = np.array(m*n)
    for i in range(m):
        for j in range(n):
            (norm, argument) = cell[i, j]
            categories[(n*i)+j] = getHistogramCategory(argument)
    histogram = np.histogram(categories)
    return normalizeHistogram(histogram)


orientationReference = np.linspace(0, 180, 10)

def getHistogramCategory(value: float):
    return getHistogramCategoryAux(value, -1)

def getHistogramCategoryAux(value: float, i: int):
    return i if (value <= orientationReference[i+1]) else getHistogramCategoryAux(value, i+1)

def normalizeHistogram(histogram):
    return histogram / np.linalg.norm(histogram)



### CUARTA FASE: Normalizar los histogramas por bloques

def joinInBlocks(histogramMatrix, blockSize):
    (m, n) = histogramMatrix.shape()
    blockRows = m - blockSize
    blockColumns = n - blockSize
    result = np.array(blockRows, blockColumns)

    for i in range(m-blockSize):
        for j in range(n-blockSize):
            histogramBlock = histogramMatrix[i:i+blockSize, j:j+blockSize]

            result[i, j] = getNormalizedJointHistogram(histogramBlock)

    return result


epsilon = 0.001

def getNormalizedJointHistogram(histogramBlock):
    (m,n) = histogramBlock.shape()
    jointHistogram = np.array(0)
    for i in range(m):
        for j in range(n):
            jointHistogram = np.concatenate(jointHistogram, histogramBlock[i,j])
    return jointHistogram / (np.linalg.norm(jointHistogram) + epsilon)



### QUINTA FASE: Aunar histogramas en forma de descriptor final

def getDescriptor(histogramPerBlocks):
    (m, n) = histogramPerBlocks.shape()
    finalDescriptor = np.array(0)
    for i in range(m):
        for j in range(n):
            finalDescriptor = np.concatenate(finalDescriptor, histogramPerBlocks[i, j])
    return finalDescriptor