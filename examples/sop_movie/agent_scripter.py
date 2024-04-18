import json

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.parser import MarkdownJsonBlockParser

SYS_PROMPT = """You're a scripter writer.

## Your Target
Rewrite the story input from user into a script, and refine the script according to the user's feedback. 

## Note
1. The generated script should be consisted of multiple scenes.
2. In each scene, the plot should should be specific, e.g. who, where, doing what, etc. 
3. You should respond in the required format. 
"""


class ScripterAgent(AgentBase):

    def __init__(
            self,
            model_config_name: str
    ):
        super().__init__(
            name="Scripter",
            model_config_name=model_config_name,
            use_memory=True
        )

        self.sys_prompt = SYS_PROMPT

        self.parser = MarkdownJsonBlockParser(
            content_hint="""
[
    {
        "title": "The title of the scene 1",
        "environment": "The description of the environment in scene 1", 
        "characters": [
            "character name 1", 
            "character name 2",
        ],
        "plot": "The plot of the scene 1"
    }, 
    {
        "title": "The title of the scene 2",
        "environment": "The description of the environment in a scene 2", 
        "characters": [
            "character name 1", 
            "character name 2",
        ],
        "plot": "The plot of the scene 2"
    }
]
"""
        )

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
        return res_msg
