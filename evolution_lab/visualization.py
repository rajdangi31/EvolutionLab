"""Real-time pygame visualization."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except ImportError:  # pragma: no cover - handled at runtime for headless users.
    pygame = None

try:
    from .agent import Agent
    from .config import SimulationConfig, VisualizationConfig
    from .simulation import Simulation
except ImportError:
    from agent import Agent
    from config import SimulationConfig, VisualizationConfig
    from simulation import Simulation


class PygameVisualizer:
    """Render the grid world and handle playback controls."""

    def __init__(self, simulation: Simulation, config: VisualizationConfig) -> None:
        if pygame is None:
            raise RuntimeError("pygame is required for visualization. Install with: pip install pygame")

        self.simulation = simulation
        self.config = config
        self.paused = False
        self.steps_per_frame = config.initial_steps_per_frame
        self.width_px = simulation.config.width * config.cell_size
        self.height_px = simulation.config.height * config.cell_size
        self.panel_height = 72

        pygame.init()
        pygame.display.set_caption("EvolutionLab")
        self.screen = pygame.display.set_mode((self.width_px, self.height_px + self.panel_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

    def run(self) -> None:
        """Run until the user closes the pygame window."""

        while True:
            self.handle_events()
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    self.simulation.step()
                    if self.simulation.step_in_generation >= self.simulation.config.generation_steps:
                        metrics = self.simulation.complete_generation()
                        print(
                            f"generation={metrics.generation} "
                            f"best={metrics.best_fitness:.2f} "
                            f"avg={metrics.average_fitness:.2f} "
                            f"diversity={metrics.diversity:.4f}"
                        )
                        self.simulation.metrics.save_csv(self.simulation.config.metrics_path)
                        self.simulation.metrics.plot(self.simulation.config.plots_dir)
            self.draw()
            self.clock.tick(self.config.fps)

    def handle_events(self) -> None:
        """Map keyboard controls to pause and speed changes."""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_UP):
                    self.steps_per_frame = min(self.config.max_steps_per_frame, self.steps_per_frame + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_DOWN):
                    self.steps_per_frame = max(self.config.min_steps_per_frame, self.steps_per_frame - 1)

    def draw(self) -> None:
        """Render food, agents, grid, and run status."""

        self.screen.fill(self.config.background_color)
        self.draw_grid()
        self.draw_food()
        self.draw_overlays()
        self.draw_agents()
        self.draw_panel()
        pygame.display.flip()

    def draw_grid(self) -> None:
        for x in range(0, self.width_px, self.config.cell_size):
            pygame.draw.line(self.screen, self.config.grid_color, (x, 0), (x, self.height_px))
        for y in range(0, self.height_px, self.config.cell_size):
            pygame.draw.line(self.screen, self.config.grid_color, (0, y), (self.width_px, y))

    def draw_food(self) -> None:
        inset = max(1, self.config.cell_size // 4)
        size = max(2, self.config.cell_size - 2 * inset)
        for x, y in self.simulation.world.food:
            pygame.draw.rect(
                self.screen,
                self.config.food_color,
                (x * self.config.cell_size + inset, y * self.config.cell_size + inset, size, size),
            )

    def draw_agents(self) -> None:
        best_agent = self.best_living_agent()
        radius = max(2, self.config.cell_size // 2 - 1)
        for agent in self.simulation.population:
            if not agent.alive:
                continue
            color = self.config.best_agent_color if agent is best_agent else self.config.agent_color
            center = (
                agent.x * self.config.cell_size + self.config.cell_size // 2,
                agent.y * self.config.cell_size + self.config.cell_size // 2,
            )
            pygame.draw.circle(self.screen, color, center, radius)

    def draw_overlays(self) -> None:
        """Show communication, internal memory, and cultural-memory usage."""

        cell = self.config.cell_size
        if self.simulation.config.communication_enabled:
            for agent in self.simulation.population:
                if not agent.alive or abs(agent.signal) < 0.35:
                    continue
                center = (agent.x * cell + cell // 2, agent.y * cell + cell // 2)
                radius = max(cell, int(abs(agent.signal) * self.simulation.config.communication_radius * cell))
                color = (120, 190, 255) if agent.signal > 0 else (255, 120, 150)
                pygame.draw.circle(self.screen, color, center, radius, width=1)

        if self.simulation.config.internal_memory_enabled:
            for agent in self.simulation.population:
                if not agent.alive or agent.memory_strength < 0.25:
                    continue
                start = (agent.x * cell + cell // 2, agent.y * cell + cell // 2)
                end = (
                    int(start[0] + agent.memory_dx * self.simulation.config.width * cell),
                    int(start[1] + agent.memory_dy * self.simulation.config.height * cell),
                )
                pygame.draw.line(self.screen, (245, 180, 70), start, end, width=1)

        if self.simulation.config.cultural_memory_enabled:
            for agent in self.simulation.population:
                if not agent.alive or agent.cultural_confidence <= 0.0:
                    continue
                center = (agent.x * cell + cell // 2, agent.y * cell + cell // 2)
                pygame.draw.circle(self.screen, (80, 230, 220), center, max(2, cell // 3), width=1)

    def draw_panel(self) -> None:
        panel_y = self.height_px
        pygame.draw.rect(self.screen, (28, 31, 37), (0, panel_y, self.width_px, self.panel_height))
        latest = self.simulation.metrics.history[-1] if self.simulation.metrics.history else None
        best = latest.best_fitness if latest else 0.0
        avg = latest.average_fitness if latest else 0.0
        status = "paused" if self.paused else "running"
        lines = [
            f"Gen {self.simulation.generation} | Step {self.simulation.step_in_generation}/{self.simulation.config.generation_steps} | {status}",
            f"Speed {self.steps_per_frame}x | Best {best:.1f} | Avg {avg:.1f} | overlays: signal/memory/culture",
        ]
        for index, line in enumerate(lines):
            text = self.font.render(line, True, self.config.text_color)
            self.screen.blit(text, (12, panel_y + 12 + index * 28))

    def best_living_agent(self) -> Agent | None:
        living_agents = [agent for agent in self.simulation.population if agent.alive]
        if not living_agents:
            return None
        return max(living_agents, key=lambda agent: agent.food_eaten * 100 + agent.energy)


def run_visualization(sim_config: SimulationConfig, viz_config: VisualizationConfig) -> None:
    """Start a pygame visualization run."""

    simulation = Simulation(sim_config)
    PygameVisualizer(simulation, viz_config).run()
