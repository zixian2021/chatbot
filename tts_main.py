import logging
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
#from text import text_to_sequence
import numpy as np
from scipy.io import wavfile
import torch
import json
import commons
import utils
import sys
import pathlib
import onnxruntime as ort
import gradio as gr
import argparse
import time
import os
import io
from scipy.io.wavfile import write
from flask import Flask, request
from threading import Thread
from transformers import AutoTokenizer, AutoModel
import openai
import requests
from scipy.io import wavfile
from text.symbols import symbols
from text import cleaned_text_to_sequence
from vits_pinyin import VITS_PinYin
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)


class VitsGradio:
    def __init__(self):
        self.lan = ["中文","日文","自动"]
        self.speaker_choice = ["刘德华","周杰伦","肖战","杨幂"]
        self.speaker_dict = {"刘德华":0,"周杰伦":1,"肖战":2,"杨幂":3}
        self.chatapi = ["gpt-3.5-turbo","gpt3"]
        self.modelPaths = []
        for root,dirs,files in os.walk("checkpoints"):
            for dir in dirs:
                self.modelPaths.append(dir)

        #self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
        #self.model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().quantize(4).cuda()
        self.history = []  
        hps = utils.get_hparams_from_file("./configs/bert_vits_star.json")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.net_g = utils.load_class(hps.train.eval_class)(
        # len(symbols),
        # hps.data.filter_length // 2 + 1,
        # hps.train.segment_size // hps.data.hop_length,
        # **hps.model)

        self.net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda(0)

        #utils.load_model("vits_bert_model.pth", self.net_g)
        utils.load_model("logs/star_vits/G_740000.pth", self.net_g)
        self.net_g.eval()
        self.net_g.to(device)


        #self.symbols = self.get_symbols_from_json(f"checkpoints/Default/config.json")
        self.symbols = symbols
        self.hps = utils.get_hparams_from_file("./configs/bert_vits.json")


        #self.hps = utils.get_hparams_from_file(f"checkpoints/Default/config.json")
        phone_dict = {
                symbol: i for i, symbol in enumerate(self.symbols)
            }
        #self.ort_sess = ort.InferenceSession(f"checkpoints/Default/model.onnx")
        self.speaker_id = 0
        self.n_scale = 0.267
        self.n_scale_w = 0.7
        self.l_scale = 1
        self.language = "中文"

        #print(self.language,self.speaker_id,self.n_scale)
        
        with gr.Blocks() as self.Vits:
            with gr.Tab("调试用"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                self.text = gr.TextArea(label="Text", value="你好")
                                with gr.Accordion(label="测试api", open=False):
                                    self.local_chat1 = gr.Checkbox(value=False, label="使用网址+文本进行模拟")
                                    self.url_input = gr.TextArea(label="键入测试", value="http://127.0.0.1:8080/chat?Text=")
                                    butto = gr.Button("模拟前端抓取语音文件")
                                btnVC = gr.Button("测试tts+对话程序")
                            with gr.Column():
                                output2 = gr.TextArea(label="回复")
                                output1 = gr.Audio(label="采样率22050")
                                output3 = gr.outputs.File(label="44100hz: output.wav")
                butto.click(self.Simul, inputs=[self.text, self.url_input], outputs=[output2,output3])
                btnVC.click(self.tts_fn, inputs=[self.text], outputs=[output1,output2])
                
                #stream = gr.State()
                
                #inp = gr.Audio(source="microphone")

                #butto.click(self.Simul, inputs=[self.text, self.url_input], outputs=[output2,output3])
                
                #gradio的流式输出

                
                #output1

                #btnVC.click(self.tts_fn, inputs=[stream], outputs=[output1,output2])

                

# define queue - required for generators
# demo.queue()
# demo.launch()
                #inp.stream(self.tts_fn, inputs=[inp, stream], outputs=[output1,output2,stream])
            with gr.Tab("控制面板"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                self.api_input1 = gr.TextArea(label="输入api-key或本地存储说话模型的路径", value="/root/autodl-tmp/chatbot/ChatGLM-6B/thudm_model")
                                with gr.Accordion(label="chatbot选择", open=False):
                                                self.api_input2 = gr.Checkbox(value=True, label="采用gpt3.5")
                                                self.local_chat1 = gr.Checkbox(value=False, label="启动本地chatbot")
                                                self.local_chat2 = gr.Checkbox(value=False, label="是否量化")
                                                res = gr.TextArea()
                                Botselection = gr.Button("完成chatbot设定")
                                Botselection.click(self.check_bot, inputs=[self.api_input1,self.api_input2,self.local_chat1,self.local_chat2], outputs = [res])
                                #self.input1 = gr.Dropdown(label = "模型", choices = self.modelPaths, value = self.modelPaths[0], type = "value")
                                #self.input2 = gr.Dropdown(label="Language", choices=self.lan, value="自动", interactive=True)
                            
                            with gr.Column():
                                btnVC = gr.Button("完成vits TTS端设定")
                                self.input3 = gr.Dropdown(label="Speaker", choices=self.speaker_choice, value=0, interactive=True)
                                self.input4 = gr.Slider(minimum=0, maximum=1.0, label="更改噪声比例(noise scale)，以控制情感", value=0.267)
                                self.input5 = gr.Slider(minimum=0, maximum=1.0, label="更改噪声偏差(noise scale w)，以控制音素长短", value=0.7)
                                self.input6 = gr.Slider(minimum=0.1, maximum=10, label="duration", value=1)
                                statusa = gr.TextArea()
                
                btnVC.click(self.create_tts_fn, inputs=[self.input3, self.input4, self.input5, self.input6], outputs = [statusa])

    def Simul(self,text,url_input):
        web = url_input + text
        res = requests.get(web)
        music = res.content
        with open('output.wav', 'wb') as code:
            code.write(music)
        file_path = "output.wav"
        return web,file_path


    def chatgpt(self,text):
        self.messages.append({"role": "user", "content": text},)
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages= self.messages)
        reply = chat.choices[0].message.content
        print(reply)
        return reply
    
    def ChATGLM(self,text):
        if text == 'clear':
            self.history = []
        response, new_history = self.model.chat(self.tokenizer, text, self.history)
        response = response.replace(" ",'').replace("\n",'.')
        
        self.history = new_history
        return response
    
    def gpt3_chat(self,text):
        call_name = "Waifu"
        openai.api_key = args.key
        identity = ""
        start_sequence = '\n'+str(call_name)+':'
        restart_sequence = "\nYou: "
        if 1 == 1:
            prompt0 = text #当期prompt
        if text == 'quit':
            return prompt0
        prompt = identity + prompt0 + start_sequence
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.0,
            stop=["\nYou:"]
        )
        return response['choices'][0]['text'].strip()
    
    def check_bot(self,api_input1,api_input2,local_chat1,local_chat2):
        if local_chat1:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(api_input1, trust_remote_code=True)
            if local_chat2:
                print('load quantized chatglm')
                self.model = AutoModel.from_pretrained(api_input1, trust_remote_code=True).half().quantize(4).cuda()
                print('load quantized success')
            else:
                print('load chatglm')
                self.model = AutoModel.from_pretrained(api_input1, trust_remote_code=True)
                print('load success')
            self.history = []
        else:
            self.messages = []
            openai.api_key = "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"
        return "Finished"
    
    def is_japanese(self,string):
        for ch in string:
            if ord(ch) > 0x3040 and ord(ch) < 0x30FF:
                return True
        return False
    
    def is_english(self,string):
        import re
        pattern = re.compile('^[A-Za-z0-9.,:;!?()_*"\' ]+$')
        if pattern.fullmatch(string):
            return True
        else:
            return False

    def get_symbols_from_json(self,path):
        assert os.path.isfile(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return data['symbols']

    def sle(self,language,text):
        text = text.replace('\n','。').replace(' ',',')
        if language == "中文":
            tts_input1 = "[ZH]" + text + "[ZH]"
            return tts_input1
        elif language == "自动":
            tts_input1 = f"[JA]{text}[JA]" if self.is_japanese(text) else f"[ZH]{text}[ZH]"
            return tts_input1
        elif language == "日文":
            tts_input1 = "[JA]" + text + "[JA]"
            return tts_input1

    def get_text(self,text,hps_ms):
        text_norm = text_to_sequence(text,hps_ms.data.text_cleaners)
        if hps_ms.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def create_tts_fn(self, input2, input3, n_scale= 0.667,n_scale_w = 0.8, l_scale = 1 ):
        #self.symbols = self.get_symbols_from_json(f"checkpoints/{path}/config.json")
        #self.hps = utils.get_hparams_from_file(f"checkpoints/{path}/config.json")
        # phone_dict = {
        #         symbol: i for i, symbol in enumerate(self.symbols)
        #     }
        #self.ort_sess = ort.InferenceSession(f"checkpoints/{path}/model.onnx")
        self.language = "中文"

        

        self.speaker_id = self.speaker_dict[input2]
        # self.n_scale = n_scale
        # self.n_scale_w = n_scale_w
        # self.l_scale = l_scale
        print(self.language,self.speaker_id,self.n_scale)
        return 'success'
    
    #流式输出
    def tts_fn(self,  text):
        t0 = time.time()
        # if self.local_chat1:
        #     print("chatgpt")
        #     text = self.chatgpt(text)
        # elif self.api_input2:
        #     print("ChATGLM")
        #     text = self.ChATGLM(text)
        # else:
        #     print("gpt3")
        #     text = self.gpt3_chat(text)

        t3 = time.time()

        gpt_time = "GPT时间："+str(t3-t0)+"s" 
        print(gpt_time)
        
        text = text
        print(text)


        speaker = torch.LongTensor(1)                                                                     
        speaker[0] = self.speaker_id
        
        print("speaker:")
        print(self.speaker_id)



        #text =self.sle(self.language,text)
        #seq = text_to_sequence(text, cleaner_names=self.hps.data.text_cleaners)
        text =self.sle(self.language,text)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tts_front = VITS_PinYin("./bert", device)
        hps = utils.get_hparams_from_file("./configs/bert_vits.json")
        phonemes, char_embeds = tts_front.chinese_to_phonemes(text)
        input_ids = cleaned_text_to_sequence(phonemes)

        speaker = speaker.to(device)



        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(device)
            
            audio = self.net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, speaker, noise_scale=0.5,
                                length_scale=1)[0][0, 0].data.cpu().float().numpy()


        # if self.hps.data.add_blank:
        #     seq = commons.intersperse(seq, 0)
        # with torch.no_grad():
        #     x = np.array([seq], dtype=np.int64)
        #     x_len = np.array([x.shape[1]], dtype=np.int64)
        #     sid = np.array([self.speaker_id], dtype=np.int64)
        #     scales = np.array([self.n_scale, self.n_scale_w, self.l_scale], dtype=np.float32)
        #     scales.resize(1, 3)
        #     ort_inputs = {
        #                 'input': x,
        #                 'input_lengths': x_len,
        #                 'scales': scales,
        #                 'sid': sid
        #             }
        #     t1 = time.time()


        #     audio = np.squeeze(self.ort_sess.run(None, ort_inputs))
        #     audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
        #     audio = np.clip(audio, -32767.0, 32767.0)



            #t2 = time.time()
            #spending_time = "推理时间："+str(t2-t1)+"s" 

            #print(spending_time)


            audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
            audio = np.clip(audio, -32767.0, 32767.0)


            bytes_wav = bytes()
            byte_io = io.BytesIO(bytes_wav)
            # wavfile.write('moe/temp1.wav',self.hps.data.sampling_rate, audio.astype(np.int16))
            # cmd = 'ffmpeg -y -i '  + 'moe/temp1.wav' + ' -ar 44100 ' + 'moe/temp2.wav'
            # os.system(cmd)    
        return (self.hps.data.sampling_rate, audio),text.replace('[JA]','').replace('[ZH]','')

        
        
        
        
        # if audio is None:
        #     return gr.update(),text.replace('[JA]','').replace('[ZH]',''), instream


        
        

        #return (self.hps.data.sampling_rate, audio),text.replace('[JA]','').replace('[ZH]','')







        
        #seq = text_to_sequence(text, cleaner_names=self.hps.data.text_cleaners)
        # if self.hps.data.add_blank:
        #     seq = commons.intersperse(seq, 0)
        # with torch.no_grad():
        #     x = np.array([seq], dtype=np.int64)
        #     x_len = np.array([x.shape[1]], dtype=np.int64)
        #     sid = np.array([self.speaker_id], dtype=np.int64)
        #     scales = np.array([self.n_scale, self.n_scale_w, self.l_scale], dtype=np.float32)
        #     scales.resize(1, 3)
        #     ort_inputs = {
        #                 'input': x,
        #                 'input_lengths': x_len,
        #                 'scales': scales,
        #                 'sid': sid
        #             }
        #     t1 = time.time()
        #     audio = np.squeeze(self.ort_sess.run(None, ort_inputs))
        #     audio *= 32767.0 / max(0.01, np.max(np.abs(audio))) * 0.6
        #     audio = np.clip(audio, -32767.0, 32767.0)
        #     t2 = time.time()
        #     spending_time = "推理时间："+str(t2-t1)+"s" 

        #     print(spending_time)
        #     bytes_wav = bytes()
        #     byte_io = io.BytesIO(bytes_wav)
        #     wavfile.write('moe/temp1.wav',self.hps.data.sampling_rate, audio.astype(np.int16))
        #     cmd = 'ffmpeg -y -i '  + 'moe/temp1.wav' + ' -ar 44100 ' + 'moe/temp2.wav'
        #     os.system(cmd)    
        # return (self.hps.data.sampling_rate, audio),text.replace('[JA]','').replace('[ZH]','')

app = Flask(__name__)
print("开始部署")
grVits = VitsGradio()

@app.route('/chat')
# def text_api():
#     message = request.args.get('Text','')

#     audio,text = grVits.tts_fn(message)
#     text = text.replace('[JA]','').replace('[ZH]','')
#     with open('moe/temp2.wav','rb') as bit:
#         wav_bytes = bit.read()
#     headers = {
#             'Content-Type': 'audio/wav',
#             'Text': text.encode('utf-8')}
#     return wav_bytes, 200, headers

def gradio_interface():
    #grVits.tts_fn.queue()

    return grVits.Vits.launch()

if __name__ == '__main__':
    api_thread = Thread(target=app.run, args=("0.0.0.0", 8080))
    gradio_thread = Thread(target=gradio_interface)
    api_thread.start()
    
    gradio_thread.start()
