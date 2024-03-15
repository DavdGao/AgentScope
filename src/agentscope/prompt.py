# -*- coding: utf-8 -*-
"""Prompt engineering module."""
from typing import Any, Optional, Union
from enum import IntEnum

from agentscope.models import OpenAIWrapper, ModelWrapperBase
from agentscope.utils.tools import to_openai_dict, to_dialog_str


class PromptType(IntEnum):
    """Enum for prompt types."""

    STRING = 0
    LIST = 1


class PromptEngine:
    """Prompt engineering module for both list and string prompt"""

    def __init__(
        self,
        model: ModelWrapperBase,
        prompt_type: Optional[PromptType] = None,
    ) -> None:
        """

        Args:
            model (`ModelWrapperBase`):
                The target model for prompt engineering.
            prompt_type (`Optional[MsgType]`, defaults to `None`):
                The type of prompt, if it is None, it will be set according to
                the model.

        Example:
            With prompt engine, we encapsulate different operations for
            string- and list-style prompt, and block the prompt engineering
            process from the user.
            As a user, you can just combine you prompt as follows.

            ```python
            # prepare the component
            system_prompt = "You're a helpful assistant ..."
            hint_prompt = "You should response in Json format."
            prefix = "assistant: "

            # initialize the prompt engine and join the prompt
            engine = PromptEngine(model)
            prompt = engine.join(system_prompt, memory.get_memory(),
            hint_prompt, prefix)
            ```
        """
        self.model = model

        if prompt_type is None:
            if isinstance(model, OpenAIWrapper):
                self.prompt_type = PromptType.LIST
            else:
                self.prompt_type = PromptType.STRING
        else:
            self.prompt_type = prompt_type

    def join(
        self,
        *args: Any,
        format_map: Optional[dict] = None,
    ) -> Union[str, list[dict]]:
        """Join prompt components according to its type. The join function can
        accept any number and type of arguments. If prompt type is
        `PromptType.STRING`, the arguments will be joined by `"\\\\n"`. If
        prompt type is `PromptType.LIST`, the string arguments will be
        converted to `Msg` from `system`.
        """
        # TODO: achieve the summarize function
        if self.prompt_type == PromptType.STRING:
            return self.join_to_str(*args, format_map=format_map)
        elif self.prompt_type == PromptType.LIST:
            return self.join_to_list(*args, format_map=format_map)
        else:
            raise RuntimeError("Invalid prompt type.")

    def join_to_str(self, *args: Any, format_map: Union[dict, None]) -> str:
        """Join prompt components to a string."""
        prompt = []
        for item in args:
            if isinstance(item, list):
                items_str = self.join_to_str(*item, format_map=None)
                prompt += [items_str]
            elif isinstance(item, dict):
                prompt.append(to_dialog_str(item))
            else:
                prompt.append(str(item))
        prompt_str = "\n".join(prompt)

        if format_map is not None:
            prompt_str = prompt_str.format_map(format_map)

        return prompt_str

    def join_to_list(self, *args: Any, format_map: Union[dict, None]) -> list:
        """Join prompt components to a list of `Msg` objects."""
        prompt = []
        for item in args:
            if isinstance(item, list):
                # nested processing
                prompt.extend(self.join_to_list(*item, format_map=None))
            elif isinstance(item, dict):
                prompt.append(to_openai_dict(item))
            else:
                prompt.append(to_openai_dict({"content": str(item)}))

        if format_map is not None:
            format_prompt = []
            for msg in prompt:
                format_prompt.append(
                    {
                        k.format_map(format_map): v.format_map(format_map)
                        for k, v in msg.items()
                    },
                )
            prompt = format_prompt

        return prompt
