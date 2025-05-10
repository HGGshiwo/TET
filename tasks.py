from abc import ABC
import re
import json

class Task(ABC):
    def prompt1(self):
        return NotImplementedError()
    
    def parse1(self, pred):
        return NotImplementedError()
    
    def prompt2(self):
        return NotImplementedError()
    
    def parse2(self, pred, **kwargs):
        return NotImplementedError()
    
    def prompt3(self):
        return NotImplementedError()
    
    def build3(self, data, data1, data2, idx=None):
        return NotImplementedError()

class Description(Task):
    def prompt1(self):
        return "Below is a question related to a video: [question]. First, identify the key objects or characters, then ask questions about their states. Note: If the original question includes actions or time points, Do not include time or action directly, convert them into questions about states or attributes. For example, if original question is What B happend after A?, the question should be what is the state of B, what is the state of A instead of including 'after' directly. The questions should be related to a single frame and answerable with simple statements.  For each line of output, first provide the object, character, or environment, followed by the corresponding question, do not reply with extra text. The number of questions should less than 5. Output exmaple:\nobject name, question\n object name, question\nobject name, question"
    
    def parse1(self, pred):
        question = [out for out in pred.split("\n") if out.strip() != ""]
        question = [q.split(",")[1].strip() for q in question][:5]
        return question
    
    def prompt2(self):
        return "Please provide a brief answer to the following questions: [question], output the answer directly without other text, if there is not enough information in the frame, just answer 'not know', seperate an with line breaks. Output Example:\n1: answer1\n2: answer2\n3: answer3"
    
    def parse2(self, pred, num):
        try:
            answer = [o.split(":")[-1].strip() for o in pred.split("\n") if o != ""]
            assert num == len(answer), f"answer: {len(answer)}, num: {num}"
        except AssertionError as e:
            if len(answer) > num:
                answer = answer[:num]
            elif len(answer) < num:
                answer.extend(["not know"] * (num - len(answer)))
        return answer
    
    def prompt3(self):
        return "You are given some language descriptions of a video. The descriptions are sparsely sampled from the videos. The description consists of several video-related questions and their corresponding answers, starting with the question and then the corresponding answer. Each answer is preceded by a corresponding frame number."
    
    def build3(self, data, data1, data2, idx):
        all = []
        for qid, question in enumerate(data1["pred"]):
            frames = []
            for i in idx:
                frames.append((i, data2[i]['pred'][qid]))
            frames = "\n".join([f"{idx_frame[0]}: {idx_frame[1]}" for idx_frame in frames])
            all.append(f"The answer of {question}:\n{frames}")
        return "\n".join(all)
    
class JSON(Task):
    output = {
        "green cup": ["green cup is in baby's hand", "green cup is on the floor"],
        "baby": ["baby hold the green cup", "baby clap proudly", "baby lay on floor", "baby picked the cup up", "baby crawl"],
        "lady": ["lady sitting down"],
    }
    
    output2 = {
        "green cup": ["green cup is in baby's hand"],
        "baby": ["baby hold the green cup", "baby is drinking"],
        "lady": ["disappear"]
    }
    
    @staticmethod
    def parse_json(pred):
        pred = pred.replace("```json", "").replace("```", "")
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            try:
                pred = pred.split("{")[1].split("}")[0]
                pred = "{" + pred + "}"
                pred = json.loads(pred)
            except Exception as e:
                return None
        return pred
    
    def prompt1(self):
        return f"There is a video-related question and option. Please splits them into descriptions of someone doing something, and finally outputs a Json string, extracting the same object as the key and the corresponding description list of the object as the value. Use full names of objects rather than pronouns in descriptions.Inputs:\nwhat did the baby do after throwing the green cup away while on the floor near the end?\nA.clap proudly\nB.the lady sitting down\nC.lay on floor\nD.	just picked it up\nE.crawl\nOutputs:\n{json.dumps(self.output)}\nInputs:\n[question]\nOutputs:\n"
    
    def parse1(self, pred):
        return self.parse_json(pred)
    
    def prompt2(self):
        return f"This is a frame in a complete video. Below is a string in Json format. The key represents an object, and the value represents the state list of the object. First, determine whether the object corresponding to the key appears in the picture. If not, the state of the object is \"disappear\". If it appears, select several states that match the picture in the state list. If there is no matching state, output the description of the object state, and finally output a json file. The key represents the object, and the value represents the current state of the object.Inputs:\n{self.output}\nOutputs:\n{self.output2}\nInputs:\n[question]\nOutputs:\n"
    
    def parse2(self, pred):
        pred = pred.replace("```json", "").replace("```", "")
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            try:
                pred = pred.split("{")[1].split("}")[0]
                pred = "{" + pred + "}"
                pred = json.loads(pred)
            except:
                return None
        return pred
    
    def prompt3(self):
        return "You are given some language descriptions of a video. Each description is a json format string, the key is the object in the frame, the value is its state. If the object is invisible, its state is 'disappear'. "
    
    def build3(self, data, data1, data2, idx):
        captions = [
            f"{i}: " + json.dumps(data2[i]["pred"])
            for i in idx
            if i in data2
        ]
        return captions
    
class Description2(Description):
    def prompt1(self):
        return "Below is a question related to a video: [question]. First, identify the key objects or characters, then ask questions about their properties. Note: If the original question includes an action or a time point, do not include the time or action directly, but convert it into a question about the state or property. For example, if the original question is What was the man's expression after he threw the cup?, the question should be what was the position of the cup, what was the man's expression, and not directly include 'after'. Questions should be related to a single frame and can be answered by simple statements.Do not use pronouns in questions, use the complete characteristics plus the name as a reference. For each line of output, first provide the object, character, or environment, followed by the corresponding question, do not reply with extra text. The number of questions should less than 5. Output exmaple:\nobject name, question\n object name, question\nobject name, question"

class Yolo(Task):
    def prompt1(self):
        return "Analyze whether the following question contains a conditional related to an event. If so, express the conditional in a declarative sentence: [question], output the condition directly without additional text. If not, output 'No'"
    
    def parse1(self, pred):
        canditate = []
        if "no" in pred.lower():
            canditate.append("no")
        if "yes" in pred.lower():
            canditate.append("yes")
        if len(canditate) == 1:
            return canditate[0]
        else:
            return None
    
    def prompt2(self, pred):
        pass
        
            