import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

verticies = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )


def Cube(pos):
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(np.add(pos, verticies[vertex]))
    glEnd()

def plotCoordinates():
    glBegin(GL_LINES)
    glVertex3fv([0.0,0.0,0.0])
    glVertex3fv([50.0,0.0,0.0])
    glEnd()

    glBegin(GL_LINES)
    glVertex3fv([0.0,0.0,0.0])
    glVertex3fv([0.0,50.0,0.0])
    glEnd()

    glBegin(GL_LINES)
    glVertex3fv([0.0,0.0,0.0])
    glVertex3fv([0.0,0.0,50.0])
    glEnd()

def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)
    glTranslatef(0,0.0, -50)
    glRotatef(45, 45, 45, 1)
    model = glGetDoublev(GL_MODELVIEW_MATRIX)
    print(model)
    new_pos = np.array([5,5,5,0]).dot(model)
    print(new_pos)
    

    

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glRotatef(45, 0, 1, 0)
        Cube([5,5,5])
        glPopMatrix()
        Cube([5,5,5])
        plotCoordinates()
        #glRotatef(0.1, 0, 1, 0)
        pygame.display.flip()
        pygame.time.wait(10)


main()
