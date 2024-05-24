import json

from agentscope.agents import UserAgent, DictDialogAgent
from agentscope.message import Msg
from agentscope.parsers import MarkdownJsonDictParser

SYSTEM_PROMPT = (
    "You're a design document generator named {name}. Your target is to "
    "generate a detailed design document for a program.\n"
    "\n"
    "## WHAT YOU SHOULD DO\n"
    "1. Determine the target of the program, and ensure the target is clear, "
    "specific, measurable and achievable.\n"
    "2. Determine the programming language according to the requirement. "
    "If it's not specified, use Python by default.\n"
    "3. Generate a design document according to the requirements.\n"
    "4. Refine the design document according to the user feedback.\n"
    "\n"
    "## NOTE\n"
    "1. The program should be modularized, and each module should have a "
    "clear purpose and functionality.\n"
    "2. Ensure with the generated modules, the program can be implemented"
    "successfully.\n"
    "3. The programming should have an entry point, and the program should be "
    "executable and testable.\n"
    "4. Respond in the required format.\n"
    """5. If you're asked to generate a JSON dictionary object, note the escape characters. For example, if you want to generate CRLF in a string, use "\\\\n" rather than "\n"."""
)

from init import *


agent = DictDialogAgent(name="Generator", sys_prompt=SYSTEM_PROMPT.format(name="Generator"), model_config_name="qwen")
user = UserAgent("User")

document_parser = MarkdownJsonDictParser(
    content_hint={
        "background": "A brief background introduction",
        "target": "The target of the program",
        "language": "The programming language",
        "modules": {
            "{module name}": "The module description, including functinality, responsibility and constraints"
        },
        "run logic": "Elaborate the running logic, that is, the call relationships between modules step-by-step. First describe the main running logic from the entry point, then describe in what situations the program will call which module, and finally describe how the program will end."
    }
)

module_parser = MarkdownJsonDictParser(
    content_hint={
        "{module name}": {
            "attributes": {
                "{attribute name}": "The attribute description, including type, default value (if has), and constraints",
            },
            "functions": {
                "{function name}": {
                    "description": "The usage of the function",
                    "arguments": {
                        "{argument name}": "The argument description, including type, default value (if has), and constraints",
                    },
                    "return": "The return value description",
                }
            },
        }
    }
)


agent.set_parser(document_parser)

x = Msg("User", """I want to achieve a Snake game in Python, the detailed requirements: 
1. Game Platform: Console-based
2. Initial Snake Length: Starts at 2 units
3. Maximum Snake Length: Unlimited growth
4. Movement Controls: WASD keys
5. Game Speed: Constant throughout the game
6. Scoring System: Included and displayed during gameplay
7. Game Over Conditions: Ends when the snake hits a wall or its own body
8. Restart Option: No restart option upon game over""", "user", echo=True)

while True:
    x = agent(x)
    # record the doc
    design_document = x.content
    x = user(x)
    if x.content == "exit":
        break

module_names = list(design_document["modules"].keys())

x = Msg(
    "system",
    content=f"""Generate detailed API documents for all modules, including {", ".join(module_names)}

## Note:
1. Each module is a class in Python.
2. You should describe the attributes, methods, and functionalities of each module.
3. The generated API documents should be clear, specific and achievable. 
""",
    role="system",
    echo=True
)

agent.set_parser(module_parser)

while True:
    x = agent(x)
    # record the module doc
    module_document = x.content

    # If agent misses some modules
    missing_modules = set(module_names) - set(module_document.keys())
    if len(missing_modules) > 0:
        missing_modules_name = ", ".join(missing_modules)
        x = Msg("system", f"Your generated API document misses the following modules: {missing_modules_name}. Regenerate the detailed API document for all modules.", "system", echo=True)
    else:
        x = user(x)
        if x.content == "exit":
            break

# Combine the design document and module document
module_descriptions = design_document["modules"]
design_document["modules"] = {}
for module_name in module_document.keys():
    design_document["modules"][module_name] = {
        "describe": module_descriptions[module_name],
    }
    design_document["modules"][module_name].update(module_document[module_name])

print(json.dumps(design_document, indent=4))