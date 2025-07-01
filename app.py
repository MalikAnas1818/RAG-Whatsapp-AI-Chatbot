from flask import Flask,request
from googlesearch import search 
from twilio.twiml.messaging_response import MessagingResponse
from agent import get_answer

app=Flask(__name__)

@app.route("/", methods=["POST"])
def bot():
    user_msg = request.values.get('Body', '').strip()
    response = MessagingResponse()

    answer = get_answer(directory = "faiss_index", question=user_msg)
    response.message(answer)

    return str(response)
if __name__ == "__main__":
    app.run()