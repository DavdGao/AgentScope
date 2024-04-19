import json

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.parser import MarkdownJsonBlockParser


SYS_PROMPT_CHARACTER = """You're a character extractor named "Extractor", your target is to extract characters from the given story and generate the character description.

## Note
1. The characters should be extracted from the story, which means its name and description should correspond to the story.
2. The character description will be used to generate the drawing, so it should be specific (e.g. age, appearance, etc)
3. In the story, the appearance of the character may change, so you should extract and describe the features of the character in the story, e.g. its looks, the hair color, etc.
4. You should respond in the required format.  
"""


class CharacterAgent(AgentBase):
    def __init__(self, model_config_name: str):
        super().__init__(name="Extractor", model_config_name=model_config_name)

        self.sys_prompt = SYS_PROMPT_CHARACTER

        self.parser = MarkdownJsonBlockParser(
            content_hint="""
[
    {
        "name": "The name of the character 1",
        "description": "The description of the character 1. This description will be used to generate the drawing, so it should be specific (e.g. age, appearance, etc.)",
    },
    {
        "name": "The name of the character 2",
        "description": "The description of the character 2. This description will be used to generate the drawing, so it should be specific (e.g. age, appearance, etc.)",
    }
]
""")

    def reply(self, x: dict = None) -> dict:
        self.memory.add(x)

        prompt = self.model.format(
            Msg("system", self.sys_prompt, "system"),
            self.memory.get_memory(),
            Msg("system", self.parser.format_instruction, "system"),
        )

        res = self.model(
            prompt,
            parse_func=self.parser.parse,
            max_retries=1,
        ).parsed

        self.speak(Msg(self.name, json.dumps(res, indent=4), "assistant"))

        res_msg = Msg(self.name, res, "assistant")
        self.memory.add(res_msg)

        return res_msg
