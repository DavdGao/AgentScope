from agentscope.agents import DictDialogAgent
from agentscope.message import Msg
from agentscope.parsers import MarkdownCodeBlockParser

SYSTEM_PROMPT = """You're a programmer named {name}. Your target is to programming a module according to the given design document.

## WHAT YOU SHOULD DO
1. Fully understand the design document.
2. Implement the module and its APIs in required programming language according to the design document.
3. Ensure your code is clean, readable, and maintainable.
4. You code should be modularized, and each module should have a clear purpose and functionality.
5. Make sure your code is executable and testable.
6. Each module is implemented in a separate file named as the module name in lowercase with .py extension. For example, the "Player" module should be implemented in player.py. 
7. Respond in the required format.
"""

from init import *

parser = MarkdownCodeBlockParser(language_name="python")

class MyDictDialogAgent(DictDialogAgent):
    def reply(self, x: dict = None) -> dict:
        self.memory.add(x)

        prompt = self.model.format(
            Msg("system", self.sys_prompt, "system"),
            self.memory.get_memory(),
            Msg("system", self.parser.format_instruction, "system")
        )

        res = self.model(prompt, parse_func=self.parser.parse)

        msg = Msg(self.name, res.text, "assistant", metadata=res.parsed)

        self.memory.add(msg)

        self.speak(msg)

        return msg


programmer = MyDictDialogAgent(
    "Programmer",
    SYSTEM_PROMPT.format(name="Programmer"),
    model_config_name="qwen",
)

design_document = {
    "background": "Snake is a classic video game concept where the player controls a snake, navigating through a bordered plane, eating food to grow in length, while avoiding collision with walls or its own tail. This project aims to implement a console-based version of the Snake game using Python.",
    "target": "Develop a console-based Snake game in Python with features including initial snake length of 2 units, constant game speed, WASD controls, scoring system, and game over conditions when the snake hits a wall or its own body. The game does not provide a restart option post-game over.",
    "language": "Python",
    "modules": {
        "GameEngine": {
            "describe": "Responsible for initializing the game, maintaining game state, updating the snake's position based on user input, and checking for game over conditions. It ensures constant speed and manages the scoring system.",
            "attributes": {
                "snake": "An instance of the Snake class, representing the player-controlled snake.",
                "food_generator": "An instance of the FoodGenerator class, responsible for spawning food.",
                "display_manager": "An instance of the DisplayManager class, handling game rendering on the console.",
                "input_handler": "An instance of the InputHandler class, managing user input for snake movement.",
                "score": "Integer, represents the current score of the player. Initialized to 0.",
                "game_over": "Boolean, indicates whether the game has ended. Initialized to False."
            },
            "functions": {
                "__init__": {
                    "description": "Initializes all components of the game.",
                    "arguments": {},
                    "return": "None"
                },
                "start_game": {
                    "description": "Sets up the initial game state and begins the main game loop.",
                    "arguments": {},
                    "return": "None"
                },
                "check_game_over": {
                    "description": "Evaluates if the game has ended due to collision with walls or the snake's body.",
                    "arguments": {},
                    "return": "None"
                },
                "update_score": {
                    "description": "Increments the score when the snake eats food.",
                    "arguments": {
                        "points": "Integer, the points to add to the score."
                    },
                    "return": "None"
                }
            }
        },
        "Snake": {
            "describe": "Handles the snake's movement, growth upon eating food, and collision checks with itself and boundaries. It keeps track of the snake's segments and updates their positions.",
            "attributes": {
                "segments": "List of tuples, each tuple represents the (x, y) coordinates of a snake segment.",
                "direction": "String, current movement direction of the snake ('up', 'down', 'left', 'right'). Initialized to 'right'.",
                "speed": "Integer, constant speed of the snake's movement. Set by the GameEngine."
            },
            "functions": {
                "__init__": {
                    "description": "Initializes the snake with a given initial length and direction.",
                    "arguments": {
                        "initial_length": "Integer, the starting length of the snake. Default is 2."
                    },
                    "return": "None"
                },
                "move": {
                    "description": "Updates the position of the snake based on its current direction.",
                    "arguments": {},
                    "return": "None"
                },
                "grow": {
                    "description": "Increases the length of the snake by adding a new segment at its head.",
                    "arguments": {},
                    "return": "None"
                },
                "check_collision": {
                    "description": "Checks if the snake has collided with itself or the boundaries.",
                    "arguments": {},
                    "return": "Boolean, True if collision detected, False otherwise."
                }
            }
        },
        "InputHandler": {
            "describe": "Listens for and processes user input for controlling the snake's direction using WASD keys. Ensures smooth control transition without conflicting inputs.",
            "attributes": {
                "allowed_keys": "Set of strings, represents the valid keys for snake movement ('w', 'a', 's', 'd')."
            },
            "functions": {
                "__init__": {
                    "description": "Sets up the allowed keys for input.",
                    "arguments": {},
                    "return": "None"
                },
                "get_input": {
                    "description": "Waits for and returns a valid user input for snake movement.",
                    "arguments": {},
                    "return": "String, the key pressed by the user."
                },
                "change_direction": {
                    "description": "Changes the snake's direction based on user input, ensuring smooth transitions.",
                    "arguments": {
                        "new_direction": "String, the intended new direction of the snake."
                    },
                    "return": "None"
                }
            }
        },
        "FoodGenerator": {
            "describe": "Randomly spawns food within the game area, excluding positions occupied by the snake. Notifies the GameEngine when food is eaten to update score and grow the snake.",
            "attributes": {
                "game_area_dimensions": "Tuple of integers, represents the dimensions of the game area (width, height).",
                "snake": "Instance of Snake class, used to avoid generating food on the snake's body."
            },
            "functions": {
                "__init__": {
                    "description": "Initializes the food generator with the game area dimensions.",
                    "arguments": {
                        "width": "Integer, width of the game area.",
                        "height": "Integer, height of the game area."
                    },
                    "return": "None"
                },
                "generate_food": {
                    "description": "Spawns a food item at a random unoccupied location on the game board.",
                    "arguments": {},
                    "return": "Tuple, the (x, y) coordinates of the generated food."
                }
            }
        },
        "DisplayManager": {
            "describe": "Renders the game state onto the console, including the snake, food, score, and game over message. It refreshes the display at each game cycle to reflect changes.",
            "attributes": {
                "game_area_dimensions": "Tuple of integers, represents the dimensions of the game area (width, height).",
                "screen_buffer": "List of lists, represents the visual state of the game area, ready for rendering."
            },
            "functions": {
                "__init__": {
                    "description": "Initializes the display manager with the game area dimensions.",
                    "arguments": {
                        "width": "Integer, width of the game area.",
                        "height": "Integer, height of the game area."
                    },
                    "return": "None"
                },
                "render_game": {
                    "description": "Draws the current state of the game including the snake, food, and score onto the console.",
                    "arguments": {},
                    "return": "None"
                },
                "render_game_over": {
                    "description": "Displays the game over message along with the final score.",
                    "arguments": {
                        "final_score": "Integer, the score at game over."
                    },
                    "return": "None"
                }
            }
        }
    },
    "run logic": "The program starts by initializing all modules within the "
                 "GameEngine. The main game loop, located in GameEngine, "
                 "begins by setting up the initial snake and displaying the "
                 "game board through DisplayManager. It then enters a loop "
                 "where it awaits user input via InputHandler. Upon receiving "
                 "valid input, it directs the Snake to move accordingly. "
                 "Simultaneously, it checks for collisions and if the snake "
                 "eats food via FoodGenerator, adjusting the score and snake "
                 "length respectively. If a game over condition is met, "
                 "GameEngine signals DisplayManager to show the game over "
                 "screen. The loop ends, and since there's no restart option, "
                 "the game simply terminates after displaying the final score."
}


CURRENT_STATE = {
    "modules": {
        "module1": {
            "state": "done",
            "file": "module1.py"
        },
        "module2": {
            "state": "done",
            "file": "module2.py"
        },
        "module3": {
            "state": "done",
            "file": "module3.py"
        },
    }
}

programmer.set_parser(parser)

file_mapping = {}
for name, module in design_document["modules"].items():
    print(f" {name} module ".center(80, "="))
    hint_msg = Msg("system", f"## Design Document\n{design_document}\n\nNow only implement the {name} module.", "system")
    x = programmer(hint_msg)

    code = x.content

    break
