import json, os, time
from openai import OpenAI


client = OpenAI(
    base_url=os.environ["TEACHER_BASE_URL"],
    api_key=os.environ["TEACHER_API_KEY"],
)


model = os.environ["TEACHER_MODEL"]


with open("distill/prompts.jsonl", encoding="utf-8") as src, \
    open("distill/sft.jsonl", "a", encoding="utf-8") as out:
    for line in src:
        item = json.loads(line)
        prompt = item["prompt"].strip()

        if not prompt:
            continue

        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=512,
            messages=[
            {"role":"system", "content":
            "你是客服助手。未知信息必须明确说明，不得编造订单状态。"},
            {"role":"user", "content": prompt},
            ],
            )
        
        answer = resp.choices[0].message.content.strip()
        if 20 <= len(answer) <= 2000: # 只是第一层过滤
            out.write(json.dumps({"messages":[
            {"role":"user", "content":prompt},
            {"role":"assistant", "content":answer},
            ]}, ensure_ascii=False) + "\n")
        time.sleep(0.2)