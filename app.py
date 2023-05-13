import pygame
import numpy as np
from nn import MyNeuralNetwork
from get_data import DataInitializerMNIST

# Define constants
CELL_SIZE = 20
GRID_SIZE = (28, 28)
X_OFFSET, Y_OFFSET = 20, 20
BUTTON_HEIGHT = 60
PREDICTON_AREA_WIDTH = 250

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

BLACK = (32, 32, 32)
WHITE = (200, 200, 200)
GRAY = (100, 100, 100)
BLUE = (0, 0, 220)
RED = (220, 0, 0)
GREEN = (0, 220, 0)
states = ["untrained", "training", "predictions"]


class DigitRecognizerGUI:
    def __init__(self):
        self.init_neural_network()
        self.nn.gradient_descent(
            epochs=100, learning_rate=0.5, batch_size=(60000 * 2) // 3, plot_acc=False
        )
        self.nn.test_accuracy_with_test_data()

        self.init_window()
        self.init_matrix()

        self.gameloop()

    def init_window(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Grid Drawing")
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.Font(None, 30)

    def init_matrix(self):
        self.matrix = [[0 for j in range(GRID_SIZE[1])] for i in range(GRID_SIZE[0])]

    def init_neural_network(self):
        self.nn = MyNeuralNetwork(DataInitializerMNIST())

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
        self.draw_predictions()
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

            # reset matrix with r
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.init_matrix()
            # draw cell
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

            # clear cell
            if pygame.mouse.get_pressed()[2]:
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
                    self.matrix[row][col] = 0

    def get_prediction(self):
        input = np.array(self.matrix).flatten()

        input = []
        matrix = np.array(self.matrix)
        for i in range(matrix.shape[0]):
            input.append(matrix[i].flatten())

        input = np.array(self.matrix).flatten().reshape(784, 1)

        Z1, A1, Z2, A2 = self.nn.forward_propagation(input)
        return A2

    def draw_predictions(self):
        prediction_array = self.get_prediction()  # A2

        # print(
        #     f"I am % {np.around(np.max(prediction_array)*100, 2)} certain that it is: ", np.argmax(A2)
        # )

        # draw main guess
        size_w = 100
        size_h = 50
        x_guess = (
            PREDICTION_ORIGIN_X + PREDICTON_AREA_WIDTH // 2 - size_w // 2 + 3 * X_OFFSET
        )
        y_guess = PREDICTION_ORIGIN_Y + PREDICTON_AREA_HEIGHT // 2 - size_h // 2
        pygame.draw.rect(self.screen, WHITE, (x_guess, y_guess, size_w, size_h))
        text = self.font.render(
            "It is: {}".format(np.argmax(prediction_array)), True, BLACK
        )
        text_rect = text.get_rect(center=(x_guess + size_w // 2, y_guess + size_h // 2))
        self.screen.blit(text, text_rect)

        x1_synapses = x_guess
        y1_synapses = y_guess + size_h // 2

        for i, pred in enumerate(prediction_array):
            # draw prediction array
            size = PREDICTON_AREA_HEIGHT // 10
            x = PREDICTION_ORIGIN_X
            y = PREDICTION_ORIGIN_Y + i * size

            pygame.draw.rect(self.screen, WHITE, (x, y, size - 2, size - 2))
            # pygame.draw.rect(self.screen, WHITE, (x, y, size, size))

            text_color = RED if i == np.argmax(prediction_array) else GRAY
            # Draw the predicted value as text
            text = self.font.render(
                "{:.2f}".format(100 * float(pred)), True, text_color
            )
            text_rect = text.get_rect(center=(x + size // 2, y + size // 2))
            self.screen.blit(text, text_rect)

            # draw synapses
            color = (0, pred * 255, 0)
            x0_synapses = x + (size - 2)
            y0_synapses = y + (size - 2) // 2
            pygame.draw.line(
                self.screen,
                color,
                (x0_synapses, y0_synapses),
                (x1_synapses, y1_synapses),
            )


app = DigitRecognizerGUI()
