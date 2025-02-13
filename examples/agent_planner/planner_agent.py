# -*- coding: utf-8 -*-
"""A ReAct-based planner agent."""
import datetime
import json
from typing import Optional, Literal, Union

from anthropic import BaseModel

from agentscope.agents import ReActAgent
from agentscope.service import (
    ServiceToolkit,
    ServiceResponse,
    ServiceExecStatus,
)

SYSTEM_PROMPT = """You're a helpful assistant named {name}.

# Target
Your target is to help users to solve their problems with the help of the provided tool functions.

# Note
1. When the problem is complex, you need to make a plan to solve it. You need to maintain the plan and update it according to your progress.
2. When the problem is simple, you can directly provide the solution.
3. A plan is consisted of a sequence of tasks. Each task has a name, a description, a target, and a list of requirements.
4. If you decide to make a plan to solve the problem, you need to strictly follow your plan and update it if necessary.
5. When you need more information, you can ask the user directly by invoking the `finish` function.
6. Clean the plan after the problem is solved."""  # noqa


class PlannerAgent(ReActAgent):
    """A ReAct-based planner agent, which can make and maintain a plan to solve
     complex problems.

    It works exactly like normal ReAct agents, but with built-in plan
    management tools, including `add_task`, `insert_task`, `delete_task`,
    `view_plan`, `modify_task`, and `change_task_status`.

    ..note:: Make sure your LLM is able to call tool functions.
    """

    class _Task(BaseModel):
        """The task class."""

        id: str
        """The task identifier."""

        name: str
        """The task name."""

        status: Literal["todo", "wip", "done"]
        """The status of the task."""

        description: str
        """The description of the task."""

        target: str
        """The target of the task."""

        requirements: list[str]
        """The requirements of the task."""

    def __init__(
        self,
        name: str,
        service_toolkit: ServiceToolkit,
        model_config_name: str,
        force_plan: bool = False,
        max_iters: int = 50,
        verbose: bool = True,
    ) -> None:
        """Initialize the planner agent.

        Args:
            name (`str`):
                The name of the agent.
            service_toolkit (`ServiceToolkit`):
                The toolkit for the agent.
            model_config_name (`str`):
                The name of the model config.
            force_plan (`bool`, defaults to `False`):
                Whether to force the agent to make a plan for each question.
            max_iters (`int`, defaults to 50):
                The maximum number of iterations for one response.
            verbose (`bool`, defaults to `True`):
                Whether to print the detailed information during reasoning and
                acting steps. If `False`, only the final response will be
                printed.
        """
        if force_plan:
            system_prompt = SYSTEM_PROMPT
        else:
            system_prompt = SYSTEM_PROMPT + "abc"

        # Add plan related functions to the toolkit
        service_toolkit.add(self.add_task)
        service_toolkit.add(self.insert_task)
        service_toolkit.add(self.delete_task)
        service_toolkit.add(self.view_plan)
        service_toolkit.add(self.modify_task)
        service_toolkit.add(self.change_task_status)

        super().__init__(
            name=name,
            model_config_name=model_config_name,
            sys_prompt=system_prompt,
            service_toolkit=service_toolkit,
            max_iters=max_iters,
            verbose=verbose,
        )

        self.plan: list[PlannerAgent._Task] = []

    def _find_task_by_id(self, task_id: str) -> Union[int, None]:
        """Find the index of the task with the given ID in the plan."""
        for i, task in enumerate(self.plan):
            if task.id == task_id:
                return i
        return None

    def _format_plan(self) -> str:
        """Format the plan to a readable string."""
        return json.dumps(
            self.plan,
            indent=4,
            ensure_ascii=False,
        )

    # %%%%%%%%%%%%%%%%%% Agent Built-in Service functions %%%%%%%%%%%%%%%%%%

    def insert_task(
        self,
        index: int,
        name: str,
        description: str,
        target: str,
        requirements: list[str] = None,
    ) -> ServiceResponse:
        """Insert a task to the current plan at the given index.

        Args:
            index (`int`):
                The index to insert the task.
            name (`str`):
                The name of the task.
            description (`str`):
                The description of the task.
            target (`str`):
                The target of the task.
            requirements (`list[str]`):
                The requirements of the task. Each requirement should be a
                string that describes the requirement clearly.
        """
        if index < 0 or index > len(self.plan):
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=(
                    f"Invalid index `{index}`. The index should be in the "
                    f"range [0, {len(self.plan)}]."
                ),
            )

        task_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        task = PlannerAgent._Task(
            id=task_id,
            name=name,
            status="todo",
            description=description,
            target=target,
            requirements=requirements,
        )
        self.plan.insert(index, task)

        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=f"""Task added successfully. The current plan:
````plan
{self._format_plan()}
````""",
        )

    def add_task(
        self,
        name: str,
        description: str,
        target: str,
        requirements: list[str] = None,
    ) -> ServiceResponse:
        """Add a task to the current plan.

        Args:
            name (`str`):
                The name of the task.
            description (`str`):
                The description of the task.
            target (`str`):
                The target of the task.
            requirements (`list[str]`):
                The requirements of the task. Each requirement should be a
                string that describes the requirement clearly.

        Returns:
            `ServiceResponse`:
                The response of the service.
        """
        return self.insert_task(
            index=len(self.plan),
            name=name,
            description=description,
            target=target,
            requirements=requirements,
        )

    def delete_task(
        self,
        task_id: str,
    ) -> ServiceResponse:
        """Delete a task from the current plan.

        Args:
            task_id (`str`):
                The ID of the task to be deleted. You can get the task ID by
                invoking the `view_plan` function.
        """
        index = None
        for i, task in enumerate(self.plan):
            if task.id == task_id:
                index = i
                break

        if index is None:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=f"Invalid task ID `{task_id}`. Check the task ID "
                f"by invoking the `view_plan` function.",
            )

        self.plan.pop(index)

        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content="""Task deleted successfully. The current plan:
```plan
{self._format_plan()}
```""",
        )

    def view_plan(self) -> ServiceResponse:
        """View the current plan."""
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=f"""The current plan:
````plan
{self._format_plan()}
```""",
        )

    def modify_task(
        self,
        task_id: str,
        new_name: Optional[str] = None,
        new_description: Optional[str] = None,
        new_target: Optional[str] = None,
        new_requirements: Optional[list[str]] = None,
    ) -> ServiceResponse:
        """Modify a specific task in the plan with the given fields.

        Args:
            task_id (`str`):
                The ID of the task to be modified. You can get the task ID by
                invoking the `view_plan` function.
            new_name (`str`, defaults to `None`):
                The new name of the task. If not modified, leave it as `None`.
            new_description (`str`, defaults to `None`):
                The new description of the task. If not modified, leave it as
                `None`.
            new_target (`str`, defaults to `None`):
                The new target of the task. If not modified, leave it as `None`
            new_requirements (`list[str]`, defaults to `None`):
                The new requirements of the task. If not modified, leave it as
                `None`.
        """
        index = None
        for i, task in enumerate(self.plan):
            if task.id == task_id:
                index = i
                break

        if index is None:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=f"Invalid task ID `{task_id}`. Check the task ID "
                f"by invoking the `view_plan` function.",
            )

        task = self.plan[index]
        new_task = PlannerAgent._Task(
            id=task.id,
            name=new_name or task.name,
            status=task.status,
            description=new_description or task.description,
            target=new_target or task.target,
            requirements=new_requirements or task.requirements,
        )
        self.plan[index] = new_task
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=f"""Task modified successfully. The new plan:
````plan
{self._format_plan()}
````""",
        )

    def change_task_status(
        self,
        task_id: str,
        status: Literal["todo", "wip", "done"],
    ) -> ServiceResponse:
        """Change the status of a specific task in the plan to the given status

        Args:
            task_id (`str`):
                The ID of the task to be modified. You can get the task ID by
                invoking the `view_plan` function.
            status (`Literal["todo", "wip", "done"]`):
                The new status of the task. It should be one of "todo", "wip",
                and "done".
        """
        index = self._find_task_by_id(task_id)
        if index is None:
            return ServiceResponse(
                status=ServiceExecStatus.ERROR,
                content=f"Invalid task ID `{task_id}`. Check the task ID "
                f"by invoking the `view_plan` function.",
            )

        self.plan[index].status = status
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=f"""Task status changed successfully. The new plan:
````plan
{self._format_plan()}
```""",
        )

    def finish(
        self,
        response: str,  # pylint: disable=unused-argument
    ) -> ServiceResponse:
        """Call this function when you finish the user's request or when you
        need to communicate with the user to obtain more information.

        Args:
            response (`str`):
               The response to the user, maybe a question or a statement.
        """

        return ServiceResponse(
            status=ServiceResponse.SUCCESS,
            content="Success",
        )
