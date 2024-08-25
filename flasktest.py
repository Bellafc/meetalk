import os
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import defs
import json
import time
import dspy
import copy
import pandas as pd
import csv
from pydub import AudioSegment
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from docx import Document
import requests
import Fasterwhisper
#import whisper
#model = whisper.load_model("large",device='cuda')


class Node:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children if children is not None else []
        self.content=""

    def add_content(self, content):
        self.content += content

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f'Node("{self.name}", "{self.content}", {self.children})'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'  # 上传文件的存放目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # 确保上传目录存在
CORS(app)
api_key = "sk-proj-v61lipjKs2xKR7PXE2KdT3BlbkFJHyVAFIm8ilz01VjmFk9Z"
gpt4o = dspy.OpenAI(model='gpt-4o-2024-05-13', api_key=api_key)
turbo = dspy.OpenAI(model="gpt-3.5-turbo", api_key=api_key)
ollama_llama3=dspy.OllamaLocal(model="llama3",max_tokens=4000,timeout_s=480)
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
dspy.settings.configure(lm=gpt4o)

stored_data=None
is_paused = False
updatedchapt=None
rootwithcontent=None
result_list=None
docfilepath=""
writtenorg=""
output=""
current_trainset=r"labels\empty.csv"
test_trainset=r"labels\empty.csv"
#current_trainset=r"C:\Users\ASUS\Desktop\meetalkcode\labels\emptytwolabels.csv"
#r"C:\Users\Balle\OneDrive\Desktop\meetalkcode\labels\emptytwolabels.csv"
#r"C:\Users\Balle\OneDrive\Desktop\meetalkcode\labels\lawfirmtwolabels.csv"
#r"C:\Users\Balle\OneDrive\Desktop\meetalkcode\labels\musiclabels.csv"
current_ws=r"writing_styles\writing style edu.csv"
#r"writing style.csv"
current_textfile=r"C:\Users\ASUS\Desktop\meetalkcode\textfiles\mock case2.txt"
# r"textfiles\Edu1.txt"
# r"C:\Users\Balle\OneDrive\Desktop\meetalkcode\textfiles\1130220_胡毓玲_錄音檔_whisper_text.txt"
current_audio=AudioSegment.from_file(r"C:\Users\ASUS\Downloads\Product sense mock interview_ fitness app for Meta (w ex-Instagram and ex-Uber PM).mp3", format="mp3")


ws=pd.read_csv(current_ws)
alltrainset = defs.create_trainset(current_trainset)
#ca=pd.read_csv(current_trainset)
alltrainembedd = defs.generate_embeddings_from_trainset(alltrainset,tokenizer,model)
print("alltrainembedd")
doc_path=r"C:\Users\ASUS\Desktop\user study meetalk\mock case 1\Mock case1 task2 20240713.docx"
with open(current_textfile, "r", encoding="utf-8") as file:
     original_text = file.read()


out = defs.segment_text(original_text,2)


def generate(root, rootwithcontent,alltrainset, alltrainembedd,auth,known,unknown):
    print("inputroot",defs.tree_to_string(root))
    global updatedchapt
    global output
    global current_audio
    global tokenizer
    global model
    answer=""
    auth=0
    known=0
    unknown=0
    # segment_length = 30000
    # segments = []
    # audio=current_audio
    # print("duration", audio.duration_seconds)
    # for start_time in range(0, int(audio.duration_seconds) * 10000, 30000):
    #     if start_time > 5000:
    #         start_time = start_time - 5000
    #     else:
    #         start_time = start_time
    #     if start_time+5000 > int(audio.duration_seconds) * 10000:
    #         print("audio ends")
    #
    #     else:
    #         end_time = start_time + segment_length
    #         segment = audio[start_time:end_time]
    #         segments.append(segment)
    #         segment.export("output" + str(start_time / 1000) + ".wav", format='wav')
    #         print(segment, str(start_time))
    #         audionew = AudioSegment.from_file("output" + str(start_time / 1000) + ".wav")
    #         print(audionew.duration_seconds)
    #         current_spk = None
    #         current_text = []
    #         response=Fasterwhisper.run("output" + str(start_time / 1000) + ".wav")
    #         if response=={'result': None}:
    #             print("response", response)
    #             response={'result': [{'text': ' ', 'start': 0.1, 'end': 9.1, 'spk': 1}]}
    #
    #         #data = json.loads(response)
    #
    #         spk0_text = []
    #         spk1_text = []
    #         for item in response['result']:
    #             if item['spk'] == 0:
    #                 spk0_text.append(item['text'].strip())
    #             elif item['spk'] == 1:
    #                 spk1_text.append(item['text'].strip())
    #
    #         group = f"spk0: {' '.join(spk0_text)} spk1: {' '.join(spk1_text)}"
    #         print("group",group)
    for group in out:
        while is_paused == True:
            time.sleep(5)
            print("paused")
        print(bool(updatedchapt is None))
        if updatedchapt is None:
            result = defs.show_node(output, root, rootwithcontent, group, answer, alltrainset, alltrainembedd, 2,tokenizer,model,auth,known,unknown)
            output, answer, auth, known, unknown = result[2], result[3], result[4], result[5], result[6]
        else:
            print("use updatedchapt",bool(updatedchapt is None))
            output=updatedchapt
            result = defs.show_node(output, root, rootwithcontent, group, answer, alltrainset, alltrainembedd, 2,tokenizer,model,auth,known,unknown)
            output, answer, auth, known, unknown = result[2], result[3], result[4], result[5], result[6]
        yield f"data:{json.dumps({'Result': output})}\n\n"
        updatedchapt = None
        print(auth,known,unknown)
    print("lenth out",len(out))
    print("auth",auth)
    print("known",known)
    print("unknown",unknown)

@app.route('/api/submitchapter', methods=['POST'])
def handle_submit_post():
    global stored_data
    stored_data = request.get_json()
    print("Received data:", stored_data)
    # 返回简单的成功响应
    return "Data received", 200

@app.route('/api/submitchapter', methods=['GET'])
def handle_submit_get():
    global stored_data
    global result_list
    global rootwithcontent
    global output
    if stored_data is None:
        return "No data available", 400
    result_list = stored_data.get('resultList', [])
    print("resultlist",result_list)
    root = defs.convert_to_tree_allocation(result_list)
    print("root",defs.tree_to_string(root))
    rootwithcontent = copy.deepcopy(root)
    auth=0
    known=0
    unknown=0
    return Response(generate(root, rootwithcontent,alltrainset, alltrainembedd,auth,known,unknown), content_type='text/event-stream')

@app.route('/api/written', methods=['GET'])
def handle_written_get():
    global rootwithcontent
    global result_list
    global writtenorg
    global current_ws
    writtenorg=defs.generate_writting(rootwithcontent, result_list,current_ws)
    return jsonify(writtenorg)

@app.route('/api/allocationdata', methods=['GET'])
def allocationdata():
    global alltrainset
    json_data="["
    for i in range(len(alltrainset)-1):
        json_data+=(defs.toJson(alltrainset[i]))+","
    json_data+=(defs.toJson(alltrainset[len(alltrainset)-1]))+"]"
    print(json_data)
    return jsonify(json_data)

@app.route('/api/wsdata', methods=['GET'])
def wsdata():
    global ws
    data_json = ws.to_json(orient='records', force_ascii=False)
    print(data_json)
    return jsonify(data_json)

@app.route('/suggestchapter', methods=['POST'])
def suggestchapter():
    global doc_path
    doc=Document(doc_path)
    structure = defs.process_document(doc)  # Now, this returns a dictionary
    answer = {
        "msg": "获取测试数据成功!",
        "code": 200,
        "data": {
            "writingstyledataList": structure['writingstyledataList'],
            "chapterList": structure["chapterList"],
            "allocationdataList": structure['allocationdataList']
        }
    }
    return json.dumps(answer, indent=2)


@app.route('/api/save-data', methods=['POST'])
def handle_save_data():
    global alltrainset
    global alltrainembedd
    data = request.get_json()
    os.makedirs(os.path.dirname(test_trainset), exist_ok=True)
    with open(test_trainset, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['current_content', 'label1', 'label2'])
        for row in data:
            writer.writerow([row['current_content'], row['label1'], row['label2']])
    alltrainset = defs.create_trainset(current_trainset)
    alltrainembedd = defs.generate_embeddings_from_trainset(alltrainset,tokenizer,model)
    print("alltrainembedd")

    return jsonify({"message": "Data saved successfully"}), 200



@app.route('/api/save-data-ws', methods=['POST'])
def handle_save_data_ws():
    global current_ws
    data = request.get_json()
    with open(current_ws, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['tag', 'input', 'participant', 'writing_goal', 'writing_format', 'your_role', 'analytical_thinking',
             'clout', 'authentic', 'emotional_tone', 'language','difference'])
        for row in data:
            writer.writerow([row['tag'], row['input'], row['participant'], row['writing_goal'], row['writing_format'],
                             row['your_role'], row['analytical_thinking'], row['clout'], row['authentic'],
                             row['emotional_tone'], row['language'],row['difference']])
    return jsonify({"message": "Writing Style Data saved successfully"}), 200

@app.route('/api/newwritingstyle', methods=['GET'])
def handle_newwritingstyle_get():
    global rootwithcontent
    global result_list
    return jsonify(defs.searchnewchapter(rootwithcontent, result_list))

@app.route('/api/updatechapterlist', methods=['POST'])
def handle_updatechapterlist_post():
    global updatedchapterliststr
    global result_list
    updatedchapterliststr = request.get_json()
    result_list = defs.updateresultlist(result_list, updatedchapterliststr)
    print("newresultlist",result_list)
    return jsonify({"message": "Updates uploaded successfully"}), 200

@app.route('/api/chapmodify', methods=['POST'] )
def handle_Chapsubmit():
    global updatedchapt
    global rootwithcontent
    global output
    global alltrainset
    global alltrainembedd
    global tokenizer
    global model
    updatedchapt= request.get_json()
    print(updatedchapt)
    print(output)
    print("updatedchapt",updatedchapt)
    print("before",alltrainset)
    alltrainset=defs.updatetrainset2(updatedchapt,output,alltrainset)
    return jsonify({"message": "File uploaded successfully"}), 200

@app.route('/api/writtenmodify', methods=['POST'] )
def handle_writtensubmit():
    global ws
    global writtenorg
    global result_list
    updatedwriting= request.get_json()
    ws=defs.updatetewritingstyle(writtenorg,updatedwriting,ws,result_list)
    return jsonify({"message": "writing updated successfully"}), 200

@app.route('/api/pause', methods=['POST'] )
def handle_pause():
    global is_paused
    is_paused = True
    return jsonify({"message": "paused"}), 200

@app.route('/api/goon', methods=['POST'] )
def handle_goon():
    global is_paused
    is_paused = False
    return jsonify({"message": "goon"}), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    global current_textfile
    global current_audio
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    print("name",file.filename)
    file.save(os.path.join('upload_audio_or_transcript', file.filename))
    if "txt" in file.filename:
        current_textfile=os.path.join('upload_audio_or_transcript', file.filename)
        print("txttxtxt")
    elif "mp3" or "wav" in file.filename:
        current_audio=os.path.join('upload_audio_or_transcript', file.filename)
        print("audioaudio")

    return jsonify({"message": "File uploaded successfully"}), 200

@app.route('/upload_doc', methods=['POST'])
def upload_doc():
    global docfilepath
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file.save(os.path.join('upload_sample_minutes', file.filename))
    docfilepath=os.path.join('upload_sample_minutes', file.filename)
    return jsonify({"message": "File uploaded successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True, port="8080",use_reloader=False)
