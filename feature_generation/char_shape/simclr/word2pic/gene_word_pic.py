import random

import pygame

import Character

pygame.init()
characters = Character.Symbol_lst()

for char in characters:
    font = pygame.font.Font('/Users/dewey/Library/Containers/1A416726-06EB-4796-BC34-7AA7C0E6BB18/Data/Documents/fonts/FZSKBXKK.TTF',100)
    rtext = font.render(char,True,(0,0,0),(255,255,255))
    pygame.image.save(rtext,f'characterImage/{char}.png')

import os

t = os.listdir('SimCLR/characterImage')
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir, 'SimCLR/characterImage')
random.shuffle(t)
cnt = 0
with open('./characterImageDocs-train.txt', 'w', encoding='utf-8') as f:
    for i in t[:-100]:
        f.write(os.path.join(current_dir,i)+" "+str(cnt))
        cnt += 1
        f.write('\n')

with open('./SimCLR/datasets/characterImageDocs-test.txt', 'w', encoding='utf-8') as f:
    for i in t[-100:]:
        f.write(os.path.join(current_dir, i) + " " + str(cnt))
        cnt += 1
        f.write('\n')
