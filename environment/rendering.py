
import pygame
import numpy as np

_screen = None
_clock = None
_FONT = None

CELL_W = 120
CELL_H = 40
MARGIN = 10


def render_env(env, mode='human'):
    global _screen, _clock, _FONT
    if _screen is None:
        pygame.init()
        _screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption('Adaptive Learning Resource Navigator')
        _clock = pygame.time.Clock()
        _FONT = pygame.font.SysFont('Arial', 16)

    _screen.fill((240, 240, 240))

    start_x = 20
    start_y = 20
    for i, name in enumerate(env.ACTIONS):
        x = start_x
        y = start_y + i * (CELL_H + 8)
        rect = pygame.Rect(x, y, 420, CELL_H)
        color = (200, 200, 200)
        if env.last_action == i:
            color = (170, 220, 170)
        pygame.draw.rect(_screen, color, rect)
        txt = _FONT.render(f"{i}: {name}", True, (0, 0, 0))
        _screen.blit(txt, (x + 8, y + 8))

    # Sidebar stats
    sx = 460
    sy = 20
    stats = [
        f"Step: {env.current_step}/{env.max_steps}",
        f"Mastery: {env.mastery:.3f}",
        f"Engagement: {env.engagement:.3f}",
        f"Fatigue: {env.fatigue:.3f}",
    ]
    for i, line in enumerate(stats):
        txt = _FONT.render(line, True, (0, 0, 0))
        _screen.blit(txt, (sx, sy + i * 24))

    
    inst = _FONT.render('This visualization is for the assignment (custom).', True, (50, 50, 50))
    _screen.blit(inst, (20, 380))

    if mode == 'human':
        pygame.display.flip()
        _clock.tick(10)
        return None
    elif mode == 'rgb_array':
      
        arr = pygame.surfarray.array3d(_screen)
        
        return np.transpose(arr, (1, 0, 2))


def cleanup():
    global _screen, _clock
    try:
        pygame.quit()
    except Exception:
        pass
    _screen = None
    _clock = None



