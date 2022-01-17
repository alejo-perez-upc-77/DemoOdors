import pygame
from audiomix3 import AudioMix
import time
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=4)

display = pygame.display.set_mode((800, 600))

test = AudioMix(0.1, 0.1, 0.2)
time.sleep(5)
print("no ha de cambiar")
test.update(0.1, 0.1)
print("cambiará en 3")
time.sleep(3)
test.update(-0.21, -0.21)
print('esperamos 60')
time.sleep(61)
test.update(-0.21, -0.21)
print('no debe cambiar en 5')
time.sleep(5)
test.update(-0.19, -0.21)
time.sleep(5)
print('no debe cambiar en 5')
test.update(-0.19, -0.14)
time.sleep(50)
print('esperamos 60')
test.update(0.21, -0.21)
print("ha debido Cambiar")
time.sleep(4)
print("debe cambiar")
test.update(-0.21, -0.21)
time.sleep(6)
test.stop()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
            exit()
    pygame.display.update()
pygame.quit()