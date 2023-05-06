import pygame

# Define constants
CELL_SIZE = 20
GRID_SIZE = (28, 28)
X_OFFSET, Y_OFFSET = 20, 20
BUTTON_HEIGHT = 60
PREDICTON_AREA_WIDTH = 400

MATRIX_AREA_WIDTH = CELL_SIZE * GRID_SIZE[1]
MATRIX_AREA_HEIGHT = CELL_SIZE * GRID_SIZE[0]

TOTAL_WIDTH = PREDICTON_AREA_WIDTH + 3 * X_OFFSET + MATRIX_AREA_WIDTH
TOTAL_HEIGHT = BUTTON_HEIGHT + 3 * Y_OFFSET + MATRIX_AREA_HEIGHT

PREDICTON_AREA_HEIGHT = TOTAL_HEIGHT - 2 * Y_OFFSET

WINDOW_SIZE = (TOTAL_WIDTH, TOTAL_HEIGHT)

# origins of the panels
MATRIX_ORIGIN_X, MATRIX_ORIGIN_Y = 2 * X_OFFSET + PREDICTON_AREA_WIDTH, Y_OFFSET
PREDICTION_ORIGIN_X, PREDICTION_ORIGIN_Y = X_OFFSET, Y_OFFSET
BUTTON_ORIGIN_X, BUTTON_ORIGIN_Y = (
    2 * X_OFFSET + PREDICTON_AREA_WIDTH,
    2 * Y_OFFSET + MATRIX_AREA_HEIGHT,
)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

states = ["untrained", "training", "predictions"]


class DigitRecognizerGUI:
    def __init__(self):
        self.init_window()
        self.init_matrix()

        self.gameloop()

    def init_window(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Grid Drawing")
        self.clock = pygame.time.Clock()
        self.running = True

    def init_matrix(self):
        self.matrix = [[0 for j in range(GRID_SIZE[1])] for i in range(GRID_SIZE[0])]

    def draw_matrix(self):
        # Draw cells
        for i in range(GRID_SIZE[0]):
            for j in range(GRID_SIZE[1]):
                if self.matrix[i][j] == 1:
                    color = BLACK
                else:
                    color = WHITE
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        MATRIX_ORIGIN_X + j * CELL_SIZE,
                        MATRIX_ORIGIN_Y + i * CELL_SIZE,
                        CELL_SIZE,
                        CELL_SIZE,
                    ),
                )
        self.draw_lines()

    def draw_lines(self):
        # vertical
        for i in range(GRID_SIZE[0] + 1):
            pygame.draw.line(
                self.screen,
                GRAY,
                (MATRIX_ORIGIN_X + i * CELL_SIZE, MATRIX_ORIGIN_Y),
                (MATRIX_ORIGIN_X + i * CELL_SIZE, MATRIX_ORIGIN_Y + MATRIX_AREA_WIDTH),
            )
        # horizontal
        for j in range(GRID_SIZE[1] + 1):
            pygame.draw.line(
                self.screen,
                GRAY,
                (MATRIX_ORIGIN_X, MATRIX_ORIGIN_Y + j * CELL_SIZE),
                (MATRIX_ORIGIN_X + MATRIX_AREA_WIDTH, MATRIX_ORIGIN_Y + j * CELL_SIZE),
            )

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_matrix()
        pygame.display.update()

    def gameloop(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(75)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if pygame.mouse.get_pressed()[0]:
                # Update matrix when mouse is clicked
                x, y = pygame.mouse.get_pos()
                is_colliding_x = (
                    MATRIX_ORIGIN_X < x and x < MATRIX_ORIGIN_X + MATRIX_AREA_WIDTH
                )
                is_colliding_y = (
                    MATRIX_ORIGIN_Y < y and y < MATRIX_ORIGIN_Y + MATRIX_AREA_HEIGHT
                )
                is_colliding = is_colliding_x and is_colliding_y
                if is_colliding:
                    col = (x - MATRIX_ORIGIN_X) // CELL_SIZE
                    row = (y - MATRIX_ORIGIN_Y) // CELL_SIZE
                    self.matrix[row][col] = 1
                    print("is colliding")

    def get_prediction(self):
        pass

    def draw_predictions(self):
        pass


app = DigitRecognizerGUI()
