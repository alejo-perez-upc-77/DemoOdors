import pygame
import random
from APP_Utilities import songs_generator

pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=4)



class AudioMix:
    content_songs, melancholic_songs, disgusting_songs, happy_songs = songs_generator()

    def __init__(self, x, y, hist, current_folder="default", current_song="default", start_second="default"):
        self.x = x
        self.y = y
        self.hist = hist
        self.current_folder = current_folder
        self.current_song = current_song
        self.update(self.x, self.y)
        self.start_second = start_second

    def play(self, folder):
        current_random = random.choice([x for x in folder if x != self.current_song])
        if self.start_second == 'default':
            pygame.mixer.music.load(current_random)
            pygame.mixer.music.play(loops=1)
            pygame.mixer.music.set_volume(1)

        else:
            pygame.mixer.music.fadeout(3000)
            pygame.mixer.music.load(current_random)
            pygame.mixer.music.play(start=self.start_second, loops=1)
            pygame.mixer.music.set_volume(1)

        self.current_folder = folder
        self.current_song = current_random

    def stop(self):
        pygame.mixer.music.stop()

    def update(self, x, y):
        self.x = x
        self.y = y
        self.start_second = int(pygame.mixer.music.get_pos() / 1000)

        # FIRST QUADRANT
        if (self.x > 0) & (self.y > 0):
            if (self.x > 0 + self.hist) & (self.y > 0 + self.hist):
                if self.current_folder != self.happy_songs:
                    if (self.start_second > 45) or (pygame.mixer.music.get_busy() == 0):

                        self.start_second = 'default'

                    self.play(self.happy_songs)

                else:
                    if pygame.mixer.music.get_busy() == 0:
                        self.start_second = 'default'
                        self.play(self.happy_songs)

            else:
                if pygame.mixer.music.get_busy() == 0:
                    self.start_second = 'default'
                    if self.current_folder == 'default':
                        self.current_folder = self.happy_songs
                    self.play(self.current_folder)
                else:
                    pass

        # SECOND QUADRANT
        if (self.x < 0) & (self.y > 0):
            if (self.x < 0 - self.hist) & (self.y > 0 + self.hist):
                if self.current_folder != self.disgusting_songs:
                    if (self.start_second > 45) or (pygame.mixer.music.get_busy() == 0):
                        self.start_second = 'default'

                    self.play(self.disgusting_songs)

                else:
                    if pygame.mixer.music.get_busy() == 0:
                        self.start_second = 'default'
                        self.play(self.disgusting_songs)

            else:
                if pygame.mixer.music.get_busy() == 0:
                    self.start_second = 'default'
                    if self.current_folder == 'default':
                        self.current_folder = self.disgusting_songs
                    self.play(self.current_folder)
                else:
                    pass

        # THIRD QUADRANT
        if (self.x < 0) & (self.y < 0):
            if (self.x < 0 - self.hist) & (self.y < 0 - self.hist):
                if self.current_folder != self.melancholic_songs:

                    if (self.start_second > 45) or (pygame.mixer.music.get_busy() == 0):
                        self.start_second = 'default'

                    self.play(self.melancholic_songs)

                else:
                    if pygame.mixer.music.get_busy() == 0:
                        self.start_second = 'default'
                        self.play(self.melancholic_songs)

            else:
                if pygame.mixer.music.get_busy() == 0:
                    self.start_second = 'default'
                    if self.current_folder == 'default':
                        self.current_folder = self.melancholic_songs
                    self.play(self.current_folder)
                else:
                    pass

        # FOURTH QUADRANT
        if (self.x > 0) & (self.y < 0):
            if (self.x > 0 + self.hist) & (self.y < 0 - self.hist):
                if self.current_folder != self.content_songs:
                    if (self.start_second > 45) or (pygame.mixer.music.get_busy() == 0):
                        self.start_second = 'default'

                    self.play(self.content_songs)

                else:
                    if pygame.mixer.music.get_busy() == 0:
                        self.start_second = 'default'
                        self.play(self.content_songs)

            else:
                if pygame.mixer.music.get_busy() == 0:
                    self.start_second = 'default'
                    if self.current_folder == 'default':
                        self.current_folder = self.content_songs
                    self.play(self.current_folder)
                else:
                    pass



