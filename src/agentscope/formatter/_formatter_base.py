from abc import ABC
from typing import Union, Sequence, List

from loguru import logger

from agentscope.message import Msg
from agentscope.utils.tools import _convert_to_str, _to_openai_image_url


class FormatterBase(ABC):
    def format(self, *args: Union[Msg, Sequence[Msg]]) -> List:
        """Format the input string and dictionary into the format that
        OpenAI Chat API required.

        Args:
            args (`Union[Msg, Sequence[Msg]]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects.
                In distribution, placeholder is also allowed.

        Returns:
            `List[dict]`:
                The formatted messages in the format that OpenAI Chat API
                required.
        """
        raise NotImplementedError


class OpenAIChatFormatter:

    support_text: bool = True
    support_images: bool = True

    substrings_in_vision_models_names = ["gpt-4-turbo", "vision", "gpt-4o"]
    """The substrings in the model names of vision models."""

    def __init__(self, model_name: str) -> None:
        """Initialize the OpenAIChatFormatter with the model name

        Args:
            model_name (`str`):
                The model name, e.g. gpt-4, gpt-4-turbo, gpt-4o, etc.
        """
        self.model_name = model_name

    def format(
            self,
            *args: Union[Msg, Sequence[Msg]],
    ) -> List[dict]:
        """Format the input string and dictionary into the format that
        OpenAI Chat API required.

        Args:
            args (`Union[Msg, Sequence[Msg]]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects.
                In distribution, placeholder is also allowed.

        Returns:
            `List[dict]`:
                The formatted messages in the format that OpenAI Chat API
                required.
        """
        messages = []
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, Msg):
                if arg.url is not None:
                    messages.append(self._format_msg_with_url(arg))
                else:
                    messages.append(
                        {
                            "role": arg.role,
                            "name": arg.name,
                            "content": _convert_to_str(arg.content),
                        },
                    )

            elif isinstance(arg, list):
                messages.extend(self.format(*arg))
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list "
                    f"of Msg objects, got {type(arg)}.",
                )

        return messages

    def _format_msg_with_url(
        self,
        msg: Msg,
    ) -> dict:
        """Format a message with image urls into openai chat format.
        This format method is used for gpt-4o, gpt-4-turbo, gpt-4-vision and
        other vision models.
        """
        # Check if the model is a vision model
        if not any(
            _ in self.model_name
            for _ in self.substrings_in_vision_models_names
        ):
            logger.warning(
                f"The model {self.model_name} is not a vision model. "
                f"Skip the url in the message.",
            )
            return {
                "role": msg.role,
                "name": msg.name,
                "content": _convert_to_str(msg.content),
            }

        # Put all urls into a list
        urls = [msg.url] if isinstance(msg.url, str) else msg.url

        # Check if the url refers to an image
        checked_urls = []
        for url in urls:
            try:
                checked_urls.append(_to_openai_image_url(url))
            except TypeError:
                logger.warning(
                    f"The url {url} is not a valid image url for "
                    f"OpenAI Chat API, skipped.",
                )

        if len(checked_urls) == 0:
            # If no valid image url is provided, return the normal message dict
            return {
                "role": msg.role,
                "name": msg.name,
                "content": _convert_to_str(msg.content),
            }
        else:
            # otherwise, use the vision format message
            returned_msg = {
                "role": msg.role,
                "name": msg.name,
                "content": [
                    {
                        "type": "text",
                        "text": _convert_to_str(msg.content),
                    },
                ],
            }

            image_dicts = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _,
                    },
                }
                for _ in checked_urls
            ]

            returned_msg["content"].extend(image_dicts)

            return returned_msg


class SingleUserMessageFormatter:
    def format(
        self,
        *args: Union[Msg, Sequence[Msg]],
    ) -> List[dict]:
        """Format the input string and dictionary into the unified format.
        Note that the format function might not be the optimal way to contruct
        prompt for every model, but a common way to do so.
        Developers are encouraged to implement their own prompt
        engineering strategies if have strong performance concerns.

        Args:
            args (`Union[MessageBase, Sequence[MessageBase]]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects.
                In distribution, placeholder is also allowed.
        Returns:
            `List[dict]`:
                The formatted messages in the format that anthropic Chat API
                required.
        """

        # Parse all information into a list of messages
        input_msgs = []
        for _ in args:
            if _ is None:
                continue
            if isinstance(_, Msg):
                input_msgs.append(_)
            elif isinstance(_, list) and all(
                isinstance(__, Msg) for __ in _
            ):
                input_msgs.extend(_)
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list "
                    f"of Msg objects, got {type(_)}.",
                )

        # record dialog history as a list of strings
        system_content_template = []
        dialogue = []
        for i, unit in enumerate(input_msgs):
            if i == 0 and unit.role == "system":
                # system prompt
                system_prompt = _convert_to_str(unit.content)
                if not system_prompt.endswith("\n"):
                    system_prompt += "\n"
                system_content_template.append(system_prompt)
            else:
                # Merge all messages into a dialogue history prompt
                dialogue.append(
                    f"{unit.name}: {_convert_to_str(unit.content)}",
                )

        if len(dialogue) != 0:
            dialogue_history = "\n".join(dialogue)

            system_content_template.extend(
                ["## Dialogue History", dialogue_history],
            )

        system_content = "\n".join(system_content_template)

        messages = [
            {
                "role": "user",
                "content": system_content,
            },
        ]

        return messages


class SystemUserMessageFormatter:

    def format(
        self,
        *args: Union[Msg, Sequence[Msg]],
    ) -> List:
        """Format the messages for general Chat API.

        In this format function, the input messages are formatted into a
        single system messages with format "{name}: {content}" for each
        message. Note this strategy maybe not suitable for all scenarios,
        and developers are encouraged to implement their own prompt
        engineering strategies.

        The following is an example:

        .. code-block:: python

            prompt = model.format(
                Msg("system", "You're a helpful assistant", role="system"),
                Msg("Bob", "Hi, how can I help you?", role="assistant"),
                Msg("user", "What's the date today?", role="user")
            )

        The prompt will be as follows:

        .. code-block:: python

            [
                {
                    "role": "system",
                    "content": "You're a helpful assistant",
                }
                {
                    "role": "user",
                    "content": (
                        "## Dialogue History\\n"
                        "Bob: Hi, how can I help you?\\n"
                        "user: What's the date today?"
                    )
                }
            ]


        Args:
            args (`Union[MessageBase, Sequence[MessageBase]]`):
                The input arguments to be formatted, where each argument
                should be a `Msg` object, or a list of `Msg` objects.
                In distribution, placeholder is also allowed.

        Returns:
            `List[dict]`:
                The formatted messages.
        """

        # Parse all information into a list of messages
        input_msgs = []
        for _ in args:
            if _ is None:
                continue
            if isinstance(_, Msg):
                input_msgs.append(_)
            elif isinstance(_, list) and all(
                    isinstance(__, Msg) for __ in _
            ):
                input_msgs.extend(_)
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list "
                    f"of Msg objects, got {type(_)}.",
                )

        messages = []

        # record dialog history as a list of strings
        dialogue = []
        for i, unit in enumerate(input_msgs):
            if i == 0 and unit.role == "system":
                # system prompt
                messages.append(
                    {
                        "role": unit.role,
                        "content": _convert_to_str(unit.content),
                    },
                )
            else:
                # Merge all messages into a dialogue history prompt
                dialogue.append(
                    f"{unit.name}: {_convert_to_str(unit.content)}",
                )

        if len(dialogue) == 0:
            return messages

        dialogue_history = "\n".join(dialogue)

        user_content_template = "## Dialogue History\n{dialogue_history}"

        messages.append(
            {
                "role": "user",
                "content": user_content_template.format(
                    dialogue_history=dialogue_history,
                ),
            },
        )

        return messages

