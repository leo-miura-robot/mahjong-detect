import os
import tempfile
import requests
from flask import Flask, request, abort
from dotenv import load_dotenv
from ultralytics import YOLO
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage

app = Flask(__name__)

load_dotenv()
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

model = YOLO("best2.pt")

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    reply_text = f"{event.message.text}"
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        for chunk in message_content.iter_content():
            tmp.write(chunk)
        image_path = tmp.name

    results = model.predict(source=image_path, conf=0.3, iou=0.3, imgsz=960)
    labels = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            labels.append(model.names[cls_id])

    order = [
        "1_man","2_man","3_man","4_man","5_man","6_man","7_man","8_man","9_man",
        "1_pin","2_pin","3_pin","4_pin","5_pin","6_pin","7_pin","8_pin","9_pin",
        "1_sou","2_sou","3_sou","4_sou","5_sou","6_sou","7_sou","8_sou","9_sou",
        "east","south","west","north","haku","hatsu","chun"
    ]

    labels_sorted = sorted(labels, key=lambda x: order.index(x) if x in order else 999)

    if labels_sorted:
        reply_text = "pai: " + " ".join(labels_sorted)
    else:
        reply_text = "ないよ"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
