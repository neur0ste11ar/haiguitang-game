import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms import Tongyi

flag1 = False
flag2 = False
story = ""
truth = ""
llm = Tongyi()

def generate(keyword):
    tokenizer = AutoTokenizer.from_pretrained("neurostellar/Qwen7B-haiguitang", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("neurostellar/Qwen7B-haiguitang", device_map="auto", trust_remote_code=True).eval()
    inputs = tokenizer(f"请根据给定的关键词，生成一个有创意且符合海龟汤特点的故事情节和真相。关键词：{keyword}。", return_tensors='pt')
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    return (tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

def trigger(message, history):
    global flag1, flag2, story, truth
    
    if message == "#真相":
        flag2 = False
        return "真相是\n" + truth
    

    if flag1:
        text = generate(message)
        story, truth = text.split("故事情节：")[1].split("真相：")
        flag1 = False
        flag2 = True
        return story + "\n请提问"
    
    
    if flag2:
        return llm.invoke(f"文本：{truth}，问题：{message}。请根据文本，回答是或者否或者不知道。")
        
        
    if message == "#海龟汤":
        flag1 = True
        return (f"海龟汤启动，请发一个关键词")
    
    return llm.invoke(message)

gr.ChatInterface(trigger).launch()
