import numpy as np
import pygame
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time

# Warehouse dimensions and possible moves
WAREHOUSE_SIZE = (10, 10)
CELL_SIZE = 50
MOVES = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

# Robot constraints
ROBOT_SPEED = 0.1  # Speed in m/s
MOVE_DELAY = 2     # Time to pause after each move in seconds
OBSTACLES = [ (0,4),(0,5), (1, 5),(0,6),(1,6),(2,6),(4,8),(4,9),(5,3),(5,4),(5,5),(5,6),(6,4),
(6,5),(7,4),(7,5),(8,4),(8,5),(9,4),(9,5),(10,4),(10,5),(1,1),(1,0),(1,2)]  # Obstacle positions
MOVEMENT_BOUNDARY = (10,10)  # Robot's allowed movement boundary

# Generate training data
def generate_training_data(num_samples=5000):
    inputs = []
    targets = []
    for _ in range(num_samples):
        start_x, start_y = np.random.randint(0, WAREHOUSE_SIZE[0]), np.random.randint(0, WAREHOUSE_SIZE[1])
        target_x, target_y = np.random.randint(0, WAREHOUSE_SIZE[0]), np.random.randint(0, WAREHOUSE_SIZE[1])

        dx, dy = target_x - start_x, target_y - start_y
        if abs(dx) > abs(dy):
            move = 1 if dx > 0 else 0  # down or up
        else:
            move = 3 if dy > 0 else 2  # right or left

        inputs.append([start_x, start_y, target_x, target_y])
        targets.append(move)

    return np.array(inputs), np.array(targets)

# Create dataset
inputs, targets = generate_training_data()
targets = to_categorical(targets, num_classes=4)

x_train, x_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Define a neural network model
model = Sequential([
    Dense(128, input_shape=(4,), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Function to predict the next move
def predict_next_move(position, target):
    input_data = np.array([position + target])
    prediction = model.predict(input_data)
    move_index = np.argmax(prediction)
    move = MOVES[move_index]
    return move

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WAREHOUSE_SIZE[1] * CELL_SIZE, WAREHOUSE_SIZE[0] * CELL_SIZE))
pygame.display.set_caption("Spidy Navigation ")

# Load images
warehouse_image_path = 'final_grid.png'
robot_image_path = 'bugg.png'

warehouse_image = pygame.image.load(warehouse_image_path).convert()
warehouse_image = pygame.transform.scale(warehouse_image, (WAREHOUSE_SIZE[1] * CELL_SIZE, WAREHOUSE_SIZE[0] * CELL_SIZE))

robot_img = pygame.image.load(robot_image_path).convert_alpha()
robot_img = pygame.transform.scale(robot_img, (CELL_SIZE, CELL_SIZE))

# Draw the grid and obstacles
def draw_grid():
    for i in range(1, WAREHOUSE_SIZE[0]):
        pygame.draw.line(screen, (200, 200, 200), (0, i * CELL_SIZE), (WAREHOUSE_SIZE[1] * CELL_SIZE, i * CELL_SIZE), 1)
    for j in range(1, WAREHOUSE_SIZE[1]):
        pygame.draw.line(screen, (200, 200, 200), (j * CELL_SIZE, 0), (j * CELL_SIZE, WAREHOUSE_SIZE[0] * CELL_SIZE), 1)

    # Draw obstacles
    for (ox, oy) in OBSTACLES:
        pygame.draw.rect(screen, (47,79,79), (oy * CELL_SIZE, ox * CELL_SIZE, CELL_SIZE, CELL_SIZE))

# Boundary and obstacle check
def keep_within_bounds(position):
    x, y = position
    x = max(0, min(MOVEMENT_BOUNDARY[0] - 1, x))
    y = max(0, min(MOVEMENT_BOUNDARY[1] - 1, y))

    # Check if the position is an obstacle
    if (x, y) in OBSTACLES:
        return None  # Return None if it's an obstacle
    return (x, y)

# Display the robot's position on the warehouse image
def display_robot_position(position, target):
    screen.blit(warehouse_image, (0, 0))
    draw_grid()

    # Draw target
    target_x, target_y = target
    pygame.draw.circle(screen, (0, 255, 0), (target_y * CELL_SIZE + CELL_SIZE // 2, target_x * CELL_SIZE + CELL_SIZE // 2), 10)

    # Draw robot
    pos_x, pos_y = position
    screen.blit(robot_img, (pos_y * CELL_SIZE, pos_x * CELL_SIZE))

    pygame.display.flip()

# Run autonomous system
start_position = (0, 0)
target_position = (7,9)
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the robot
    move = predict_next_move(list(start_position), list(target_position))
    new_position = (start_position[0] + move[0], start_position[1] + move[1])

    # Check if the new position is within bounds and not an obstacle
    checked_position = keep_within_bounds(new_position)

    if checked_position is not None:
        start_position = checked_position
    else:
        print("Move blocked by obstacle or out of bounds, Searching for alternative route")
        # Try alternative moves if the predicted move is blocked
        for move_index, alternative_move in MOVES.items():
            new_position = (start_position[0] + alternative_move[0], start_position[1] + alternative_move[1])
            checked_position = keep_within_bounds(new_position)
            
            if checked_position is not None:
                start_position = checked_position
                break
    # Update display
    display_robot_position(start_position, target_position)
    pygame.display.flip()

    # Simulate robot speed and mandatory pause
    time.sleep(ROBOT_SPEED)
    pygame.time.delay(int(MOVE_DELAY * 1000))

    # Check if reached target
    if start_position == target_position:
        print("Reached destination!")
        pygame.time.delay(1000)
        break

pygame.quit()
