from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.parsers import MarkdownCodeBlockParser

SYSTEM_PROMPT="""You're a programmer named {name}. Your target is to programming a module according to the design document.

## WHAT YOU SHOULD DO
1. Fully understand the design document.
2. Implement the module and its APIs in required programming language according to the design document.
3. Ensure your code is clean, readable, and maintainable.
4. You code should be modularized, and each module should have a clear purpose and functionality.
5. Make sure your code is executable and testable.
5. Respond in the required format.
"""

class ProgrammerAgent(AgentBase):

    def __init__(
        self,
        model_config_name: str
    ) -> None:

        name = "Programmer"
        super().__init__(
            name=name,
            sys_prompt=SYSTEM_PROMPT.format(name=name),
            model_config_name=model_config_name,
            use_memory=True,
        )

        self.memory.add(Msg("system", self.sys_prompt, "system"))

        self.parser = MarkdownCodeBlockParser(
            language_name="python",
        )

    def reply(self, x: dict = None) -> dict:
        self.memory.add(x)

        prompt = self.model.format(
            self.memory.get_memory(),
            Msg("system", self.parser.format_instruction, "system"),
        )

        res = self.model(
            prompt,
            parse_func=self.parser.parse
        )

        msg = Msg(self.name, res.parsed, "assistant")
        self.speak(msg)

        self.memory.add(msg)

        return msg


from init import *

agent = ProgrammerAgent(model_config_name="qwen")

modules = {'GameBoard': {'attributes': {'width': 'The width of the game board in characters.', 'height': 'The height of the game board in characters.', 'snake_segments': "List of tuples representing the (x, y) positions of the snake's body segments.", 'food_position': 'A tuple representing the (x, y) position of the food item.', 'board_state': 'A 2D list representing the visual state of the game board.'}, 'methods': {'initialize_board': {'description': 'Sets up the initial state of the game board, including placing the snake and food.', 'arguments': {}, 'return': 'None'}, 'render_board': {'description': 'Prints the current state of the game board to the console.', 'arguments': {}, 'return': 'None'}, 'update_board': {'description': "Updates the game board state based on the snake's movement and food consumption.", 'arguments': {}, 'return': 'None'}, 'detect_collision': {'description': "Checks for collisions with walls or the snake's body.", 'arguments': {}, 'return': 'Boolean indicating if a collision has occurred.'}}}, 'SnakeControl': {'attributes': {'direction': 'Current movement direction of the snake (up, down, left, right).'}, 'methods': {'change_direction': {'description': "Changes the snake's direction based on user input, ensuring valid transitions.", 'arguments': {'new_direction': 'The new direction input by the user.'}, 'return': 'None'}, 'get_next_move': {'description': "Determines the next position of the snake's head based on the current direction.", 'arguments': {}, 'return': 'Tuple representing the (x, y) coordinates of the next move.'}}}, 'SnakeGrowth': {'attributes': {'snake_segments': "Reference to the GameBoard's snake_segments attribute to manage the snake's body."}, 'methods': {'grow_snake': {'description': "Extends the snake's body by adding a new segment at its tail when it consumes food.", 'arguments': {}, 'return': 'None'}, 'check_growth': {'description': 'Verifies if the snake has eaten food and triggers growth if so.', 'arguments': {'current_head': 'The current head position of the snake.'}, 'return': 'Boolean indicating if the snake has grown.'}}}, 'ScoringSystem': {'attributes': {'score': "Integer representing the player's current score."}, 'methods': {'increment_score': {'description': 'Increases the score by a predefined amount when the snake eats food.', 'arguments': {}, 'return': 'None'}, 'display_score': {'description': 'Prints the current score to the console.', 'arguments': {}, 'return': 'None'}}}, 'GameOverHandler': {'attributes': {}, 'methods': {'check_game_over': {'description': 'Evaluates if the game has ended due to a collision.', 'arguments': {'collision_detected': 'Boolean from detect_collision indicating a collision status.'}, 'return': 'Boolean indicating if the game is over.'}, 'end_game': {'description': 'Displays the final score and terminates the game loop.', 'arguments': {}, 'return': 'None'}}}}


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

file_mapping = {}
for name, module in modules.items():
    print(module)
    # Implement the module and its APIs in required programming language according to the design document.
    hint_msg = Msg("system", f"Implement the {name} module in Python: {module}", "system")

    x = agent(hint_msg)
