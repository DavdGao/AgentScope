import json

from agentscope.agents import AgentBase
from agentscope.message import Msg
from agentscope.parser import MarkdownJsonBlockParser

SYS_PROMPT_STORYBOARD = """You're a storyboard generator named "Generator". Your target is to generate storyboard (a sequence of drawings, typically with some directions and dialogue, representing the shots planned for a movie or television show) according to the given scene script and character descriptions. 

## Note
1. You should decide how many drawings are needed for a given scene. These drawings should be important and meaningful in the scene.
3. Each drawing should be a key frame in this scene or story, which should be meaningful and important. 
4. Your description for each drawing will be used to generate a shot image. So the description should be specific. 
5. Adjust your response according to the user's feedback.
6. You should respond in the required format. 
"""


class StoryboardAgent(AgentBase):

    def __init__(
            self,
            model_config_name: str
    ):
        super().__init__(
            name="Generator",
            model_config_name=model_config_name,
            use_memory=True
        )

        self.sys_prompt = SYS_PROMPT_STORYBOARD

        self.parser = MarkdownJsonBlockParser(
            content_hint="""
[
    {
        "title": "The title of drawing 1", 
        "description": "The description of drawing 1, which should describe an image of the scene",
    },
    {
        "title": "The title of drawing 2", 
        "description": "The description of drawing 2, which should describe an image of the scene",
    }
]
""",
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
        self.memory.add(res_msg)

        return res_msg
