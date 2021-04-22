import numpy as np
import matplotlib.image as img
from matplotlib import pyplot as plt
from PIL import Image

#load anh tu tep 
def loadImage(nombreImage = 'image.jpg', mostrarImagen = True):
  img = Image.open(nombreImage)
  if (mostrarImagen == True):
    img.show()
  return img

#convert hinh anh
#dua vao mang
def convertirImagen(imagen, mostrarImagen = False):
  #RGB
  #LA
  #CMYK
  #YCbCr
  #HSV
  imagen_gris = imagen.convert('LA')
  imagen_gris_matriz = np.array(list(imagen_gris.getdata(band=0)), float)
  imagen_gris_matriz.shape = imagen_gris.size[1], imagen_gris.size[0]
  imagen_matriz = np.matrix(imagen_gris_matriz)
  if (mostrarImagen == True): 
    plt.imshow(imagen_matriz, cmap='gray')
    plt.show()
  return imagen_matriz


def SVD(imagen_matriz):
  #KHAI THÁC GIÁ TRỊ SINGULAR
  U_svd, sigma, V_svd = np.linalg.svd(imagen_matriz)
  n = 50 #N càng cao thì độ phân giải càng cao
  frobenius = []
  for i in range(3, n+1, 5):
    #nhân ma trận U * D * V để xây dựng lại ma trận ban đầu
    #n quyết định số phần tử cần lấy từ các ma trận.
    U =  np.matrix(U_svd[:, :i])
    D =  np.diag(sigma[:i]) #lấy đường chéo
    V =  np.matrix(V_svd[:i, :])
    imagen_reconstruida =  U*D*V 
    plt.imshow(imagen_reconstruida, cmap='gray')
    frob = (np.linalg.norm(imagen_matriz) - np.linalg.norm(imagen_reconstruida))
    titulo = "n = " + str(i) + " norma frobenius " + str(frob)
    frobenius.append(frob)
    plt.title(titulo)
    plt.show()
  x = range(len(frobenius))
  plt.plot(x, frobenius, 'ro')
  plt.show()


imagen = loadImage('image.jpg')
matriz = convertirImagen(imagen)
SVD(matriz)
