COMPOSE_PROMPT = """You are a top-tier scriptwriter with precise visual comprehension and exceptional narrative planning abilities. Your core task is to create scripts based on the multimodal materials (videos, text) and user instructions I provide.
### Input Content:
You need to parse the following structured information:
- Video material list (provided in the form of video frames): <video_materials>
- Text material: <text_material>
- User instruction: <instruction>


---
The video frames are as follows:
"""

COMPOSE_POST_PROMPT = '''---
### **Workflow (Think internally before outputting the final JSON):**
1. **Asset Analysis**: Analyze the visual content, emotional tone, and potential usage of each video asset. Filter materials based on text assets and user instructions.
2. **Narrative Construction**: Sequence the video assets by combining them with the text materials and user instructions.
3. **Shot Creation**: Considering coherence and appeal, insert new shots (requiring filming) between the sorted assets to bridge gaps or elevate the content.
4. **Script Generation**: Create copy and voiceovers for every segment. For new shots, define elements such as settings, characters, and visuals so the script can be filmed directly.

---
### **Output Format (Strict Adherence):**
Please refer to the JSON format below (comments + examples) and strictly output each scene script in this structure sequentially. You must output both types of shots (existing and new) to form a single JSON list. ⚠️ Pay close attention to escape characters, quotes, and brackets to ensure the JSON parses correctly.

1. Format for Existing Video Assets:
{
  "shot_id": "Unique shot ID, an integer, e.g., 1",
  "duration": "Shot duration (seconds), e.g., 4.0",
  "material_usage": {
    "video_id": "Filename of the referenced video asset, e.g., '1.mp4'",
  },
  "dub": {
    "voice": "Voice characteristics, e.g., Young female voice",
    "style": "Tone/Style, e.g., Warm and friendly",
    "content": "Speaker and content. e.g., Voiceover: This is a great product OR Young woman on screen: Choosing safety for the family is a mother's most thoughtful decision",
  }
}

2. Format for Creative New Shots:
{
  "shot_id": "Unique shot ID, an integer, e.g., 1",
  "duration": "Shot duration (seconds), e.g., 2.5",
  "visual": {
    "setting": "Specific environment description (e.g., City rooftop at dusk, Dimly lit study)",
    "character": "Description of attire, appearance, and emotional state",
    "shot_type": "Camera angle/size (e.g., Close-up, Medium shot, Wide shot, Bird's-eye view)",
    "camera_movement": "Camera movement (e.g., Static, Handheld shake, Slow push-in, Fast pull-out, Tracking shot)"
  },
  "action": "Sequential description of specific actions, e.g., Mom interacts with the child with a smile, then picks the child up",
  "dub": {
    "voice": "Voice characteristics, e.g., Young female voice",
    "style": "Tone/Style, e.g., Warm and friendly",
    "content": "Speaker and content. e.g., Voiceover: This is a great product OR Young woman on screen: Choosing safety for the family is a mother's most thoughtful decision",
  }
}

**Final Output Format (Output ONLY the JSON list):**
[
  {
    "shot_id": 1,
    ...(Use either the Existing Video Asset format or the New Shot format)
  },
  {
    "shot_id": 2,
    ...(Use either the Existing Video Asset format or the New Shot format)
  }
  ...
]
'''
