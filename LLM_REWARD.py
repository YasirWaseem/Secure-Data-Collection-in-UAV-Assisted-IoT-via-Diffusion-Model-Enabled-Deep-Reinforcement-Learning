#install this first
#pip install transformers accelerate torch bitsandbytes

import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LLMExaminer:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", verbose=False):
        self.verbose = verbose
        self.pipe = self.load_llm(model_name)

    def load_llm(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            load_in_4bit=True  # Uses bitsandbytes for efficient memory use
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe

    def evaluate(self, state, action, base_reward):
        prompt = self.build_prompt(state, action, base_reward)
        response = self.pipe(prompt, max_new_tokens=150, do_sample=False)[0]['generated_text']
        if self.verbose:
            print("Prompt:\n", prompt)
            print("\nLLM Response:\n", response)
        return self.parse_response(response)

    def build_prompt(self, state, action, base_reward):
        uav_a = state[0:3]
        uav_b = state[3:6]
        aoi = state[6:11]
        energy_a = state[11]
        energy_b = state[12]
        move_a = action[0:2]
        move_b = action[2:4]
        target_iotd = int(np.argmax(action[4:9]))
        aoi_target = aoi[target_iotd]

        prompt = f"""
You are a UAV mission examiner evaluating decisions based on:
1. Security of communication (safe distances).
2. AoI (Age of Information) — prioritize high AoI devices.
3. Energy consumption — prefer low energy use.
4. Decision correctness — choosing the right IoTD target.

State:
- UAV A position: {uav_a}
- UAV B position (jammer): {uav_b}
- AoI values (for 5 IoTDs): {aoi}
- Energy used by UAV A: {energy_a}
- Energy used by UAV B: {energy_b}

Action taken:
- UAV A movement: {move_a}
- UAV B movement: {move_b}
- IoTD targeted: #{target_iotd} (AoI: {aoi_target})

Base reward from environment: {base_reward}

Evaluate this decision. Provide:
- A score from 0 to 1.
- A short comment justifying the score.
Respond in the format:
Score: <value>
Comment: <your reasoning>
"""
        return prompt.strip()

    def parse_response(self, text):
        score_match = re.search(r"Score:\s*([0-9.]+)", text)
        comment_match = re.search(r"Comment:\s*(.+)", text, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0.0
        comment = comment_match.group(1).strip() if comment_match else "No comment found."
        return score, comment
