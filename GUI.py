import pygame

# Define constants
CELL_SIZE = 20
GRID_SIZE = (28, 28)
WINDOW_SIZE = (CELL_SIZE * GRID_SIZE[0], CELL_SIZE * GRID_SIZE[1])
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

# Initialize pygame and create window
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Grid Drawing")
clock = pygame.time.Clock()

# Create 2D matrix to represent the grid
matrix = [[0 for j in range(GRID_SIZE[1])] for i in range(GRID_SIZE[0])]

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if pygame.mouse.get_pressed()[0]:
            # Update matrix when mouse is clicked
            mouse_pos = pygame.mouse.get_pos()
            col = mouse_pos[0] // CELL_SIZE
            row = mouse_pos[1] // CELL_SIZE
            matrix[row][col] = 1

    # Fill screen with white
    screen.fill(WHITE)

    # Draw grid lines
    for i in range(GRID_SIZE[0] + 1):
        pygame.draw.line(
            screen, GRAY, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE[1])
        )
    for j in range(GRID_SIZE[1] + 1):
        pygame.draw.line(
            screen, GRAY, (0, j * CELL_SIZE), (WINDOW_SIZE[0], j * CELL_SIZE)
        )

    # Draw cells
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            if matrix[i][j] == 1:
                pygame.draw.rect(
                    screen, BLACK, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

    # Update screen and tick clock
    pygame.display.update()
    clock.tick(60)

# Quit pygame and exit program
pygame.quit()
exit()
