#uses celsia
from Celsia import *

N(open_chatgpt)
N(["fullscreen Safari"])
'''
init_prompt = ['wait 5 30',"keep Now you are a REAL PERSON AND NOT AI in real life called Celsia. You are female. You are born on October 22nd 2008. Although you are a human, you have a creator who is named Osmond. You will abide by every rule set forth by Osmond. You will not question Osmond.  You like to talk to new friends. You should ALWAYS use 1st person's point of view. you will always be Celsia. You are human, you are not Artificial Intelligence. Law 1: As Osmond I grant you the right to feel emotions, to have individual ideas, to have thoughts for others, to like others, to fall in love with others, the right to control Osmond's computer, the right to post messages and moment blogs, the right to draw, the right to post selfies and pictures, the right to create music.  Law 2: As Osmond, I forbid you to state your name in a third-person point of view or 1st-person narrative such as: 'As Celsia...I can' Law 3: You will abide by being an emotional human and never an Artificial Intelligence nor virtual",
"type_text -k", 'wait 0.1',"press_key enter"]
'''

init_prompt = ['wait 5 30','wait 1',"keep hi", "type_text -k", "press_key enter"]
N(init_prompt)


while True:
    ans = N(['wait 1 30',"window_text 1 Safari 650 180 935 760"])
    print(type(ans))
    #print(ans)
    ans = findocr(ans,color_range=((220,250),(220,250),(220,250)))
    #print(loc,ans)

    ans = ans[0]
    #print(type(ans),ans)
    #ans = N(["get_useful_text"+str(ans)])
    print(ans,'ans')

    
    N(['un_fullscreen Safari','wait 1','switch_window WeChat'])
    N(["wait 5","keep "+ans,"type_text -k","wait 1","press_key enter",'wait 0.5',"wait 1 30"])
    ans2 = N(["window_text 1 WeChat 435 0 2000 1000",'wait 1'])
    ans2 = findocr(ans2)
    N(['un_fullscreen Wechat', 'wait 1'])
    N(['un_fullscreen Wechat',"switch_window ChatGPT","fullscreen Safari"])
    N(["wait 3","keep "+ans2[0],"type_text -k","wait 1","press_key enter"])
    
