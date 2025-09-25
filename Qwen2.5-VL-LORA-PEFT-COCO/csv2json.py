import pandas as pd
import json

# 载入CSV文件
df = pd.read_csv('./coco-2024-dataset.csv')
conversations = []

# 添加对话数据
for i in range(len(df)):
    conversations.append({
        "image": f"{df.iloc[i]['image_path']}",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nDescribe this image."
            },
            {
                "from": "gpt", 
                "value": df.iloc[i]['caption']
            }
        ]
    })

# 保存为Json
with open('data_vl.json', 'w', encoding='utf-8') as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)