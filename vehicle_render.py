import pygame
import numpy as np
import time
import pygame
from pygame import gfxdraw, freetype
class Renderer(object):

    def __init__(self, vehicle_length = None, trailer_length = None):
        self.screen_dim = 1200
        self.screen = None
        self.clock = None
        self.isopen = True
        self.metadata = {"render_fps": 30,}
        self.render_mode = "human"
        if vehicle_length:
            self.vehicle_length = vehicle_length
        else:
            self.vehicle_length = 2.4
        if trailer_length:
            self.trailer_length = trailer_length
        else:
            self.trailer_length = 2.

    def set_state(self, state):
        self.state = state

    # def render(self):
    #     import pygame
    #     from pygame import gfxdraw
    #
    #     if self.screen is None:
    #         pygame.init()
    #         pygame.display.init()
    #         self.screen = pygame.display.set_mode(
    #             (self.screen_dim, self.screen_dim)
    #         )
    #     if self.clock is None:
    #         self.clock = pygame.time.Clock()
    #
    #     self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
    #     self.surf.fill((255, 255, 255))
    #
    #     bound = 20.0
    #     scale = self.screen_dim / (bound * 2)
    #     offset = self.screen_dim // 2
    #
    #
    #     vehicle_length = self.vehicle_length * scale
    #     vehicle_width = 0.3 * self.vehicle_length * scale
    #
    #     l, r, t, b = 0, vehicle_length, vehicle_width / 2, -vehicle_width / 2
    #     vehicle_coords = [(l, b), (l, t), (r, t), (r, b)]
    #     transformed_coords = []
    #     for c in vehicle_coords:
    #         c = pygame.math.Vector2(c).rotate_rad(self.state[2])
    #         c = (c[0] + scale * self.state[0] + offset, c[1] + scale * self.state[1] + offset)
    #         transformed_coords.append(c)
    #     gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
    #     gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))
    #
    #     trailer_dif = (pygame.math.Vector2((self.trailer_length * scale, 0))
    #                    .rotate_rad(self.state[2] + self.state[3]))
    #
    #     trailer_length = self.trailer_length * scale
    #     trailer_width = 0.3 * self.vehicle_length * scale
    #     l, r, t, b = 0, trailer_length, trailer_width / 2, -trailer_width / 2
    #     trailer_coords = [(l, b), (l, t), (r, t), (r, b)]
    #     transformed_coords = []
    #     for c in trailer_coords:
    #         c = pygame.math.Vector2(c).rotate_rad(self.state[2] + self.state[3])
    #         c = (c[0] + scale * self.state[0] + offset - trailer_dif[0],
    #              c[1] + scale * self.state[1] + offset - trailer_dif[1])
    #         transformed_coords.append(c)
    #     gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
    #     gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))
    #
    #     # rod_length = 1 * scale
    #     # rod_width = 0.2 * scale
    #     # l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
    #     # coords = [(l, b), (l, t), (r, t), (r, b)]
    #     # transformed_coords = []
    #     # for c in coords:
    #     #     c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
    #     #     c = (c[0] + offset, c[1] + offset)
    #     #     transformed_coords.append(c)
    #
    #
    #     # gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
    #     # gfxdraw.filled_circle(
    #     #     self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
    #     # )
    #
    #     # rod_end = (rod_length, 0)
    #     # rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
    #     # rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
    #     # gfxdraw.aacircle(
    #     #     self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    #     # )
    #     # gfxdraw.filled_circle(
    #     #     self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    #     # )
    #
    #     # fname = path.join(path.dirname(__file__), "assets/clockwise.png")
    #     # img = pygame.image.load(fname)
    #     # if self.last_u is not None:
    #     #     scale_img = pygame.transform.smoothscale(
    #     #         img,
    #     #         (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
    #     #     )
    #     #     is_flip = bool(self.last_u > 0)
    #     #     scale_img = pygame.transform.flip(scale_img, is_flip, True)
    #     #     self.surf.blit(
    #     #         scale_img,
    #     #         (
    #     #             offset - scale_img.get_rect().centerx,
    #     #             offset - scale_img.get_rect().centery,
    #     #         ),
    #     #     )
    #
    #     # drawing axle
    #     # gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
    #     # gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
    #
    #     self.surf = pygame.transform.flip(self.surf, False, True)
    #     self.screen.blit(self.surf, (0, 0))
    #     if self.render_mode == "human":
    #         pygame.event.pump()
    #         self.clock.tick(self.metadata["render_fps"])
    #         pygame.display.flip()
    #
    #     # else:  # mode == "rgb_array":
    #     #     return np.transpose(
    #     #         np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
    #     #     )
    def render(self):

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            freetype.init()
            self.screen = pygame.display.set_mode(
                (self.screen_dim, self.screen_dim)
            )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 20.0
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        vehicle_length = self.vehicle_length * scale
        vehicle_width = 0.3 * self.vehicle_length * scale
        steering_length = vehicle_length / 4
        steering_width = vehicle_width / 6

        def draw_vehicle(vehicle_length, vehicle_width, state, filled = True, steering = None):
            """
            state: x, y, theta
            """
            l, r, t, b = 0, vehicle_length, vehicle_width / 2, -vehicle_width / 2
            vehicle_coords = [(l, b), (l, t), (r, t), (r, b)]
            transformed_coords = []
            for c in vehicle_coords:
                c = pygame.math.Vector2(c).rotate_rad(state [2])
                c = (c [0] + scale * state [0] + offset, c [1] + scale * state [1] + offset)
                transformed_coords.append(c)
            # draw.polygon(self.surf, transformed_coords, (204, 77, 77))

            gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
            if filled:
                gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

            if steering:
                steering_x1 = 0.5 * (transformed_coords [-2] [0] + transformed_coords [-1] [0])
                steering_y1 = 0.5 * (transformed_coords [-2] [1] + transformed_coords [-1] [1])
                l, r, t, b = -steering_length, steering_length, steering_width / 2, - steering_width / 2
                coords = [(l, b), (l, t), (r, t), (r, b)]
                transformed_coords = []
                for c in coords:
                    c = pygame.math.Vector2(c).rotate_rad(state [2] + steering)
                    c = (c [0] + steering_x1, c [1] + steering_y1)
                    transformed_coords.append(c)
                gfxdraw.aapolygon(self.surf, transformed_coords, (0, 0, 0))
                # if filled:
                gfxdraw.filled_polygon(self.surf, transformed_coords, (0, 0, 0))

        draw_vehicle(vehicle_length, vehicle_width, self.state, steering=self.state [5])
        draw_vehicle(vehicle_length, vehicle_width, [0, 0, 0], filled=False)

        trailer_dif = (pygame.math.Vector2((self.trailer_length, 0))
                       .rotate_rad(self.state [2] + self.state [3]))

        trailer_state = np.array([self.state [0] - trailer_dif [0],
                                  self.state [1] - trailer_dif [1],
                                  self.state [2] + self.state [3]])

        trailer_length = self.trailer_length * scale
        trailer_width = 0.3 * self.vehicle_length * scale
        draw_vehicle(trailer_length, trailer_width, trailer_state)
        draw_vehicle(trailer_length, trailer_width, [-trailer_length, 0, 0], filled=False)

        self.surf = pygame.transform.flip(self.surf, False, True)
        # text1 = f"Time: {self.state[-1]:.2f}"
        text2 = f"Velocity: {self.state [-2]:.3f}"
        text3 = f"Steering: {self.state [-1]:.3f}"
        text_color = pygame.Color('dodgerblue')
        font = freetype.SysFont("Arial", 48)
        # font.render_to(self.surf, (50, 50), text1, text_color)
        font.render_to(self.surf, (50, 98), text2, text_color)
        font.render_to(self.surf, (50, 146), text3, text_color)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata ["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        # if self.save_video:
        #     frame = pygame.surfarray.array3d(self.surf)
        #
        #     # frame = np.flip(frame, axis=1)
        #     frame = np.transpose(frame, (1, 0, 2))  # Transpose the frame
        #     self.frames.append(frame)
        #
        #     self.frame_count += 1

def test_rendering():
    renderer = Renderer()

    for i in range(100):
        renderer.set_state((5 / 100 * i, 1, 30 / 180 * np.pi, 0 / 180 * np.pi,))
        renderer.render()
        time.sleep(0.01)

if __name__ == '__main__':
    test_rendering()