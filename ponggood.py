#!/usr/bin/env python3
# Based on https://python101.readthedocs.io/pl/latest/pygame/pong/#
import pygame
from typing import Type
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as fuzzcontrol
import math

FPS = 30


class Board:
    def __init__(self, width: int, height: int):
        self.surface = pygame.display.set_mode((width, height), 0, 32)
        pygame.display.set_caption("AIFundamentals - PongGame")

    def draw(self, *args):
        background = (0, 0, 0)
        self.surface.fill(background)
        for drawable in args:
            drawable.draw_on(self.surface)

        pygame.display.update()


class Drawable:
    def __init__(self, x: int, y: int, width: int, height: int, color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.color = color
        self.surface = pygame.Surface(
            [width, height], pygame.SRCALPHA, 32
        ).convert_alpha()
        self.rect = self.surface.get_rect(x=x, y=y)

    def draw_on(self, surface):
        surface.blit(self.surface, self.rect)


class Ball(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        radius: int = 20,
        color=(255, 10, 0),
        speed: int = 3,
    ):
        super(Ball, self).__init__(x, y, radius, radius, color)
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed = speed
        self.y_speed = speed
        self.start_speed = speed
        self.start_x = x
        self.start_y = y
        self.start_color = color
        self.last_collision = 0

    def bounce_y(self):
        self.y_speed *= -1

    def bounce_x(self):
        self.x_speed *= -1

    def bounce_y_power(self):
        self.color = (
            self.color[0],
            self.color[1] + 10 if self.color[1] < 255 else self.color[1],
            self.color[2],
        )
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed *= 1.1
        self.y_speed *= 1.1
        self.bounce_y()

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.x_speed = self.start_speed
        self.y_speed = self.start_speed
        self.color = self.start_color
        self.bounce_y()

    def move(self, board: Board, *args):
        self.rect.x += round(self.x_speed)
        self.rect.y += round(self.y_speed)

        if self.rect.x < 0 or self.rect.x > (
            board.surface.get_width() - self.rect.width
        ):
            self.bounce_x()

        if self.rect.y < 0 or self.rect.y > (
            board.surface.get_height() - self.rect.height
        ):
            self.reset()

        timestamp = pygame.time.get_ticks()
        if timestamp - self.last_collision < FPS * 4:
            return

        for racket in args:
            if self.rect.colliderect(racket.rect):
                self.last_collision = pygame.time.get_ticks()
                if (self.rect.right < racket.rect.left + racket.rect.width // 4) or (
                    self.rect.left > racket.rect.right - racket.rect.width // 4
                ):
                    self.bounce_y_power()
                else:
                    self.bounce_y()


class Racket(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 80,
        height: int = 20,
        color=(255, 255, 255),
        max_speed: int = 10,
    ):
        super(Racket, self).__init__(x, y, width, height, color)
        self.max_speed = max_speed
        self.surface.fill(color)

    def move(self, x: int, board: Board):
        delta = x - self.rect.x
        delta = self.max_speed if delta > self.max_speed else delta
        delta = -self.max_speed if delta < -self.max_speed else delta
        delta = 0 if (self.rect.x + delta) < 0 else delta
        delta = (
            0
            if (self.rect.x + self.width + delta) > board.surface.get_width()
            else delta
        )
        self.rect.x += delta


class Player:
    def __init__(self, racket: Racket, ball: Ball, board: Board) -> None:
        self.ball = ball
        self.racket = racket
        self.board = board

    def move(self, x: int):
        self.racket.move(x, self.board)

    def move_manual(self, x: int):
        pass

    def act(self, x_diff: int, y_diff: int):
        pass


class PongGame:
    def __init__(
        self, width: int, height: int, player1: Type[Player], player2: Type[Player]
    ):
        pygame.init()
        self.board = Board(width, height)
        self.fps_clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)

        self.opponent_paddle = Racket(x=width // 2, y=0)
        self.oponent = player1(self.opponent_paddle, self.ball, self.board)

        self.player_paddle = Racket(x=width // 2, y=height - 20)
        self.player = player2(self.player_paddle, self.ball, self.board)

    def run(self):
        while not self.handle_events():
            self.ball.move(self.board, self.player_paddle, self.opponent_paddle)
            self.board.draw(
                self.ball,
                self.player_paddle,
                self.opponent_paddle,
            )
            self.oponent.act(
                self.oponent.racket.rect.centerx - self.ball.rect.centerx,
                self.oponent.racket.rect.centery - self.ball.rect.centery,
            )
            self.player.act(
                self.player.racket.rect.centerx - self.ball.rect.centerx,
                self.player.racket.rect.centery - self.ball.rect.centery,
            )
            self.fps_clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                return True
        keys = pygame.key.get_pressed()
        if keys[pygame.constants.K_LEFT]:
            self.player.move_manual(0)
        elif keys[pygame.constants.K_RIGHT]:
            self.player.move_manual(self.board.surface.get_width())
        return False


class NaiveOponent(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(NaiveOponent, self).__init__(racket, ball, board)

    def act(self, x_diff: int, y_diff: int):
        x_cent = self.ball.rect.centerx
        self.move(x_cent)


class HumanPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(HumanPlayer, self).__init__(racket, ball, board)

    def move_manual(self, x: int):
        self.move(x)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------


# =========================
# Hybrid Fuzzy–Predictive Edge AI (Mamdani)
# =========================
class FuzzyPlayer(Player):
    """
    Predict contact-x (with wall reflections). Use Mamdani fuzzy logic to decide
    the edge-bias gain k in [0,1] based on time-to-contact and alignment error.
    Then aim for: target = contact_x + sign(ball_x_speed) * k * EDGE_OFFSET,
    capped by how far we can actually move before contact (reachability).
    """

    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(FuzzyPlayer, self).__init__(racket, ball, board)

        # Geometry / constants
        self.W = float(self.board.surface.get_width())
        self.H = float(self.board.surface.get_height())
        self.PW = float(self.racket.width)
        self.HW = self.PW / 2.0
        self.V = float(self.racket.max_speed)

        # Edge target depth (outer ~25% band)
        self.EDGE_FRAC = 0.78
        self.EDGE_OFFSET = self.EDGE_FRAC * self.HW

        # Safety margins
        self.WALL_MARGIN = self.HW + 4.0

        # --------- Fuzzy system for edge gain k ---------
        # Inputs:
        #   t = time-to-contact (frames, >= 0)
        #   e = abs(x_err) = |(contact_x - paddle_center_x)|
        # Output:
        #   k in [0,1] (how much of EDGE_OFFSET to apply)
        T_MAX = max(30.0, self.H / max(1.0, abs(self.ball.y_speed))) * 2.0  # generous cap
        E_MAX = self.W / 2.0  # worst-case misalignment

        # Define Fuzzy inupts
        self.t = fuzzcontrol.Antecedent(np.linspace(0.0, T_MAX, 301), 't')
        self.e = fuzzcontrol.Antecedent(np.linspace(0.0, E_MAX, 301), 'e')
        self.k = fuzzcontrol.Consequent(np.linspace(0.0, 1.0, 201), 'k')

        # Memberships for time-to-contact
        self.t['short'] = fuzz.trimf(self.t.universe, [0.0, 0.0, T_MAX*0.25])
        self.t['med']   = fuzz.trimf(self.t.universe, [T_MAX*0.15, T_MAX*0.40, T_MAX*0.65])
        self.t['long']  = fuzz.trapmf(self.t.universe, [T_MAX*0.50, T_MAX*0.70, T_MAX, T_MAX])

        # Memberships for alignment error
        self.e['small'] = fuzz.trimf(self.e.universe, [0.0, 0.0, self.W*0.06])
        self.e['med']   = fuzz.trimf(self.e.universe, [self.W*0.04, self.W*0.12, self.W*0.20])
        self.e['large'] = fuzz.trapmf(self.e.universe, [self.W*0.16, self.W*0.28, E_MAX, E_MAX])

        # Output k (gain)
        self.k['low']   = fuzz.trimf(self.k.universe, [0.0, 0.0, 0.35])
        self.k['mid']   = fuzz.trimf(self.k.universe, [0.25, 0.55, 0.8])
        self.k['high']  = fuzz.trapmf(self.k.universe, [0.65, 0.85, 1.0, 1.0])

        rules = [
            fuzzcontrol.Rule(self.t['long']  & self.e['small'], self.k['high']),
            fuzzcontrol.Rule(self.t['med']   & self.e['small'], self.k['high']),
            fuzzcontrol.Rule(self.t['med']   & self.e['med'],   self.k['mid']),
            fuzzcontrol.Rule(self.t['long']  & self.e['med'],   self.k['mid']),
            fuzzcontrol.Rule(self.t['short'] & self.e['small'], self.k['mid']),
            fuzzcontrol.Rule(self.t['short'] & (self.e['med'] | self.e['large']), self.k['low']),
            fuzzcontrol.Rule(self.t['med']   & self.e['large'], self.k['low']),
            fuzzcontrol.Rule(self.t['long']  & self.e['large'], self.k['low']),
        ]

        self._sys = fuzzcontrol.ControlSystem(rules)

    # ---------- physics helpers ----------
    def _predict_contact_x_and_dt(self) -> tuple[float, float]:
        paddle_cy = float(self.racket.rect.centery)
        ball_cy   = float(self.ball.rect.centery)
        vy = float(self.ball.y_speed)

        if vy <= 0:
            return float(self.ball.rect.centerx), 0.0

        dt = (paddle_cy - ball_cy) / vy
        if dt <= 0:
            return float(self.ball.rect.centerx), 0.0

        xmin = 0.0 + self.ball.width / 2.0
        xmax = self.W - self.ball.width / 2.0
        x    = float(self.ball.rect.centerx)
        vx   = float(self.ball.x_speed)
        t    = dt

        while t > 0:
            if vx > 0:
                time_to_wall = (xmax - x) / vx
            elif vx < 0:
                time_to_wall = (x - xmin) / (-vx)
            else:
                return x, dt
            if time_to_wall >= t:
                x += vx * t
                break
            else:
                x += vx * time_to_wall
                vx = -vx
                t -= time_to_wall

        return float(np.clip(x, xmin, xmax)), float(max(0.0, dt))

    def _clamp_center(self, x_center: float) -> float:
        return float(np.clip(x_center, self.WALL_MARGIN, self.W - self.WALL_MARGIN))

    # ---------- main policy ----------

    # Sit under the ball
    def act(self, x_diff: int, y_diff: int):
        if self.ball.y_speed <= 0:
            target_x = self._clamp_center(float(self.ball.rect.centerx))
            return self.move(int(target_x))

        # Where and when will the ball land
        contact_x, dt = self._predict_contact_x_and_dt()
        cur_x = float(self.racket.rect.centerx)
        x_err = contact_x - cur_x
        e_val = abs(x_err)

        # Running Mandani fuzzy logic
        sim = fuzzcontrol.ControlSystemSimulation(self._sys)
        sim.input['t'] = float(dt)
        sim.input['e'] = float(e_val)
        try:
            sim.compute()
            k_fuzzy = float(sim.output['k'])
            if not np.isfinite(k_fuzzy):
                k_fuzzy = 0.0
        except Exception:
            k_fuzzy = 0.0

        reachable = max(0.0, self.V * dt)
        if abs(x_err) >= reachable:
            k_cap = 0.0
        else:
            extra = reachable - abs(x_err)
            k_cap = max(0.0, min(1.0, extra / max(1e-6, self.EDGE_OFFSET)))

        k_final = max(0.0, min(1.0, min(k_fuzzy, k_cap)))

        prefer_right = (self.ball.x_speed >= 0)
        edge_off = (self.EDGE_OFFSET * k_final) * (1.0 if prefer_right else -1.0)

        target = contact_x + edge_off
        clamped = self._clamp_center(target)
        if abs(target - clamped) > 1e-6 and k_final > 0.0:
            edge_off = -edge_off
            target = contact_x + edge_off
            target = self._clamp_center(target)
        else:
            target = clamped

        self.move(int(target))


# =========================
# TSK (Sugeno) Fuzzy–Predictive Edge AI
# =========================
class TSKFuzzyPlayer(Player):
    """
    Zero-order TSK controller for the same predictive policy as above.
    Inputs: time-to-contact t, alignment error e = |contact_x - paddle_x|.
    Output: k in [0,1] (edge gain), computed via weighted average of rule consequents.
    """

    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(TSKFuzzyPlayer, self).__init__(racket, ball, board)

        # Geometry / constants
        self.W = float(self.board.surface.get_width())
        self.H = float(self.board.surface.get_height())
        self.PW = float(self.racket.width)
        self.HW = self.PW / 2.0
        self.V = float(self.racket.max_speed)

        # Edge target depth (outer ~25% band)
        self.EDGE_FRAC = 0.78
        self.EDGE_OFFSET = self.EDGE_FRAC * self.HW

        # Safety margins
        self.WALL_MARGIN = self.HW + 4.0

        # Universe caps (mirrors Mamdani setup)
        self.T_MAX = max(30.0, self.H / max(1.0, abs(self.ball.y_speed))) * 2.0
        self.E_MAX = self.W / 2.0

        # Predefine rule consequents (zero-order TSK constants for k)
        # High, mid, low levels chosen to mirror Mamdani intent.
        self.K_LOW  = 0.18
        self.K_MID  = 0.52
        self.K_HIGH = 0.92

    # ---------- tiny MF helpers (scalar) ----------
    @staticmethod
    def _trimf(x: float, a: float, b: float, c: float) -> float:
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / max(1e-9, (b - a))
        return (c - x) / max(1e-9, (c - b))

    @staticmethod
    def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return (x - a) / max(1e-9, (b - a))
        return (d - x) / max(1e-9, (d - c))

    # ---------- physics helpers ----------
    def _predict_contact_x_and_dt(self) -> tuple[float, float]:
        paddle_cy = float(self.racket.rect.centery)
        ball_cy   = float(self.ball.rect.centery)
        vy = float(self.ball.y_speed)

        if vy <= 0:
            return float(self.ball.rect.centerx), 0.0

        dt = (paddle_cy - ball_cy) / vy
        if dt <= 0:
            return float(self.ball.rect.centerx), 0.0

        xmin = 0.0 + self.ball.width / 2.0
        xmax = self.W - self.ball.width / 2.0
        x    = float(self.ball.rect.centerx)
        vx   = float(self.ball.x_speed)
        t    = dt

        while t > 0:
            if vx > 0:
                time_to_wall = (xmax - x) / vx
            elif vx < 0:
                time_to_wall = (x - xmin) / (-vx)
            else:
                return x, dt
            if time_to_wall >= t:
                x += vx * t
                break
            else:
                x += vx * time_to_wall
                vx = -vx
                t -= time_to_wall

        return float(np.clip(x, xmin, xmax)), float(max(0.0, dt))

    def _clamp_center(self, x_center: float) -> float:
        return float(np.clip(x_center, self.WALL_MARGIN, self.W - self.WALL_MARGIN))

    # ---------- fuzzy inference (TSK zero-order) ----------
    def _tsk_k(self, t: float, e: float) -> float:
        T_MAX, E_MAX = self.T_MAX, self.E_MAX

        # Time-to-contact membership
        mu_t_short = self._trimf(t, 0.0, 0.0, T_MAX * 0.25)
        mu_t_med   = self._trimf(t, T_MAX * 0.15, T_MAX * 0.40, T_MAX * 0.65)
        mu_t_long  = self._trapmf(t, T_MAX * 0.50, T_MAX * 0.70, T_MAX, T_MAX)

        # Error membership
        mu_e_small = self._trimf(e, 0.0, 0.0, self.W * 0.06)
        mu_e_med   = self._trimf(e, self.W * 0.04, self.W * 0.12, self.W * 0.20)
        mu_e_large = self._trapmf(e, self.W * 0.16, self.W * 0.28, E_MAX, E_MAX)

        # Rule firing strengths (product t-norm)
        def AND(a, b): return a * b
        def OR(a, b):  return max(a, b)

        w = []
        z = []

        # If long & small -> HIGH
        w.append(AND(mu_t_long, mu_e_small)); z.append(self.K_HIGH)
        # If med & small -> HIGH
        w.append(AND(mu_t_med,  mu_e_small)); z.append(self.K_HIGH)

        # If med & med -> MID
        w.append(AND(mu_t_med,  mu_e_med));   z.append(self.K_MID)
        # If long & med -> MID
        w.append(AND(mu_t_long, mu_e_med));   z.append(self.K_MID)

        # If short & small -> MID
        w.append(AND(mu_t_short, mu_e_small)); z.append(self.K_MID)

        # If short & (med or large) -> LOW
        w.append(AND(mu_t_short, OR(mu_e_med, mu_e_large))); z.append(self.K_LOW)
        # If med  & large -> LOW
        w.append(AND(mu_t_med,   mu_e_large)); z.append(self.K_LOW)
        # If long & large -> LOW
        w.append(AND(mu_t_long,  mu_e_large)); z.append(self.K_LOW)

        num = sum(w_i * z_i for w_i, z_i in zip(w, z))
        den = sum(w) + 1e-12
        k = num / den
        if not np.isfinite(k):
            k = 0.0
        return float(np.clip(k, 0.0, 1.0))

    # ---------- main policy ----------
    def act(self, x_diff: int, y_diff: int):
        # A) If ball moving away, sit under it
        if self.ball.y_speed <= 0:
            target_x = self._clamp_center(float(self.ball.rect.centerx))
            return self.move(int(target_x))

        # B) Predictive target + TSK gain
        contact_x, dt = self._predict_contact_x_and_dt()
        cur_x = float(self.racket.rect.centerx)
        x_err = contact_x - cur_x
        e_val = abs(x_err)

        k_fuzzy = self._tsk_k(float(dt), float(e_val))

        # Reachability cap
        reachable = max(0.0, self.V * dt)
        if abs(x_err) >= reachable:
            k_cap = 0.0
        else:
            extra = reachable - abs(x_err)
            k_cap = max(0.0, min(1.0, extra / max(1e-6, self.EDGE_OFFSET)))

        k_final = max(0.0, min(1.0, min(k_fuzzy, k_cap)))

        prefer_right = (self.ball.x_speed >= 0)
        edge_off = (self.EDGE_OFFSET * k_final) * (1.0 if prefer_right else -1.0)

        target = contact_x + edge_off
        clamped = self._clamp_center(target)
        if abs(target - clamped) > 1e-6 and k_final > 0.0:
            edge_off = -edge_off
            target = contact_x + edge_off
            target = self._clamp_center(target)
        else:
            target = clamped

        self.move(int(target))


# =========================
# Entry point with a simple switch
# =========================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pong AI (Mamdani / TSK)")
    parser.add_argument("--ai", choices=["mamdani", "tsk"], default="tsk",
                        help="Bottom paddle AI type (default: tsk)")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=400)
    args = parser.parse_args()

    BottomAI = FuzzyPlayer if args.ai == "mamdani" else TSKFuzzyPlayer

    # Top paddle: naive; Bottom paddle: chosen AI
    game = PongGame(args.width, args.height, NaiveOponent, BottomAI)
    game.run()
