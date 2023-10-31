from flask import Flask, request, jsonify
from model.model_call import recommend, getEmotion

app = Flask(__name__)


@app.route('/music', methods=['POST'])
def sentiment():
    data = request.get_json()
    emotion = data.get('emotion')
    content = data.get('content')
    musicList = recommend(emotion, content)
    return jsonify(
        {
            "musicList": musicList
        }
    )


@app.route('/emotion', methods=['POST'])
def emotion():
    data = request.get_json()
    content = data.get('content')
    emotion = getEmotion(content)
    return jsonify(
        {
            "emotion": emotion
        }
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
