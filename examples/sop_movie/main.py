import agentscope
from agentscope.agents import UserAgent
from agentscope.message import Msg
from agentscope.models import load_model_by_config_name
from agentscope.service import download_from_url
from examples.sop_movie.agent_character import CharacterAgent
from examples.sop_movie.agent_scripter import ScripterAgent
from examples.sop_movie.agent_shoter import StoryboardAgent

TEST_STORY = """This is a true story about a coal miner who works in a small coal mine. He survived a gas leak that claimed his brother's life and spent 14 days in the ICU, narrowly escaping death multiple times. Despite the loss of his youngest son in a motorcycle accident and the hardships of supporting his family by working long hours in the mine, the man never complains. He cherishes his loving wife and remaining child. As their 10th wedding anniversary approaches, he wishes to surprise his wife with a gold bracelet but lacks the funds. He lingers in a bracelet shop daily, hoping to find a way. One rainy day, the shop owner notices him and, after learning his story, is moved. They share a heartfelt conversation, and the man leaves the store with gratitude and hope. On the anniversary, the man brings his wife to the shop where the owner presents them with the bracelet and organizes a surprise celebration. As his wife wears the bracelet, her joy is evident, marking a moment of warmth and kindness."""

agentscope.init(model_configs=[
    {
        "model_type": "dashscope_chat",
        "config_name": "gpt-4",

        "model_name": "qwen-max",
        "api_key": "sk-7cee068707fe4885890ee272c8b14175"
    },
    {
        "model_type": "post_api_chat",
        "config_name": "dall-e",

        "api_url": "http://47.88.8.18:8088/v1/images/create_image",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": "Bearer eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjIyNTE4NiIsInBhc3N3b3JkIjoiMjI1MTg2IiwiZXhwIjoyMDA2OTMzNTY1fQ.wHKJ7AdJ22yPLD_-1UHhXek4b7uQ0Bxhj_kJjjK0lRM",
        },
        "json_args": {
            "model": "dall-e-3",
            "n": 1,
            "size": "1024x1024"
        },
        "messages_key": "prompt"
    }
])

# model
dall_e_model = load_model_by_config_name("dall-e")

# agent
user = UserAgent(name="User")
character_agent = CharacterAgent(model_config_name="gpt-4")
scriptwriter_agent = ScripterAgent(model_config_name="gpt-4")
storyboard_agent = StoryboardAgent(model_config_name="gpt-4")


# Step 1: Extract characters from the story
x = Msg("user", TEST_STORY, "user")
while True:
    x = character_agent(x)
    x = user(x)
    if x.content == "exit":
        # record the generated characters
        generated_characters = x.content
        break

# Step 2: Split the story into scenes and generate scripts for each scene
x = Msg("user", {"story": TEST_STORY, "characters": generated_characters}, "user", echo=True)
while True:
    x = scriptwriter_agent(x)
    x = user(x)
    if x.content == "exit":
        # record the current script
        generated_script = x.content
        break


# Step 3: Generate several storyboard shots for each scene
for i_scene, scene in enumerate(generated_script):
    print(f" Scene {i_scene} ".center(80, "#"))

    msg = Msg("system", {"scene": scene, "character descriptions": generated_characters}, "system")
    while True:
        msg = storyboard_agent(msg)
        # record current shot description
        generated_shots = msg.content
        # obtain user feedback
        msg = user(msg)
        if msg.content == "exit":
            # Clear the memory for the last scene
            storyboard_agent.memory.clear()

            for i_shot, shot in enumerate(generated_shots):
                # generate an image for the scene
                img_url = dall_e_model().image_urls
                # download image from url
                download_from_url(img_url, f"scene-{i_scene}_shot-{i_shot}.png")

            break
